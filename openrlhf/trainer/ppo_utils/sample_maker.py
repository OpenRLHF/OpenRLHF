import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import ray
import torch
from tqdm import tqdm

from openrlhf.trainer.ppo_utils.experience_maker import Experience
from openrlhf.trainer.ppo_utils.filter_hooks import DynamicFilteringHook, FilterHookBase, NoOpFilterHook
from openrlhf.trainer.ppo_utils.misc import build_prompt_dataloader
from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)


# https://github.com/vllm-project/vllm/blob/main/vllm/utils/__init__.py#L41
def random_uuid() -> str:
    MASK_64_BITS = (1 << 64) - 1
    return f"{uuid.uuid4().int & MASK_64_BITS:016x}"  # 16 hex chars


def _collect_prompts(dataloader_iter, num_prompts: int):
    """Draw up to `num_prompts` items from the prompt dataloader."""
    prompts, labels, metadatas = [], [], []
    exhausted = False

    while len(prompts) < num_prompts:
        try:
            rand_metadatas, rand_prompts, rand_labels = next(dataloader_iter)
            remaining = num_prompts - len(prompts)
            prompts.extend(rand_prompts[:remaining])
            labels.extend(rand_labels[:remaining])
            metadatas.extend(rand_metadatas[:remaining])
        except StopIteration:
            exhausted = True
            break

    return prompts, labels, metadatas, exhausted


def _build_sampling_params(args, **kwargs):
    from vllm import SamplingParams

    return SamplingParams(
        temperature=kwargs.get("temperature", 1.0),
        top_p=kwargs.get("top_p", 1.0),
        top_k=kwargs.get("top_k", -1),
        max_tokens=kwargs.get("max_new_tokens", 1024),
        min_tokens=kwargs.get("min_new_tokens", 1),
        skip_special_tokens=kwargs.get("skip_special_tokens", False),
        logprobs=1 if args.enable_vllm_is_correction else None,
    )


def _build_sample_from_output(output, truncate_length, **kwargs):
    """Turn a single vLLM response into a Sample."""
    tokenized_observation = output["observation_tokens"].copy()
    tokenized_ranges = output["action_ranges"]
    metadata = output.get("metadata")
    reward_val = output.get("reward", None)
    score_val = output.get("scores", None)

    sequences = torch.tensor(tokenized_observation, dtype=torch.long)
    attention_mask = torch.tensor([1] * len(tokenized_observation))

    action_mask = torch.zeros_like(attention_mask)
    for start, end in tokenized_ranges:
        action_mask[start:end] = 1

    sequences = sequences[:truncate_length].to("cpu")
    attention_mask = attention_mask[:truncate_length].to("cpu")
    action_mask = action_mask[1:truncate_length].to("cpu")
    if output["rollout_log_probs"] is not None:
        rollout_log_probs = torch.tensor(output["rollout_log_probs"][1:truncate_length]).to("cpu")
    else:
        rollout_log_probs = None

    ones_indices = torch.where(action_mask)[0]
    response_length = (ones_indices[-1] - ones_indices[0] + 1).item() if len(ones_indices) else 0
    total_length = attention_mask.float().sum()
    is_clipped = total_length >= truncate_length

    info = {
        "response_length": torch.tensor([response_length]),
        "total_length": torch.tensor([total_length]),
        "response_clip_ratio": torch.tensor([is_clipped]),
    }
    if reward_val is not None:
        info["reward"] = torch.tensor([reward_val])
    if score_val is not None:
        info["score"] = torch.tensor([score_val])

    extra_logs = output.get("extra_logs", {})
    for key, value in extra_logs.items():
        if isinstance(value, torch.Tensor):
            value = value.flatten()[0].item()
        elif hasattr(value, "item"):
            try:
                value = value.item()
            except Exception:
                pass
        info[key] = torch.tensor([value])

    return Sample(
        sequences=sequences.unsqueeze(0),
        attention_mask=attention_mask.unsqueeze(0),
        action_mask=action_mask.unsqueeze(0),
        rollout_log_probs=rollout_log_probs.unsqueeze(0) if rollout_log_probs is not None else None,
        prompts=[output["prompt"]],
        labels=[output["label"]],
        rewards=torch.tensor([reward_val]) if reward_val is not None else None,
        scores=torch.tensor([score_val]) if score_val is not None else None,
        info=info,
        metadata=metadata,
    )


@dataclass
class Sample:
    """Lightweight container for rollout outputs prior to model inference."""

    sequences: torch.Tensor
    attention_mask: torch.Tensor
    action_mask: torch.Tensor
    rollout_log_probs: Optional[torch.Tensor]
    prompts: List[str]
    labels: List[str]
    rewards: torch.Tensor
    scores: torch.Tensor
    info: dict
    metadata: Optional[Any] = None
    index: Optional[List[int]] = None  # filled by trainer before batching

    def to_experience(self) -> Experience:
        """Convert a Sample to an Experience for downstream processing."""
        return Experience(
            index=self.index,
            sequences=self.sequences,
            attention_mask=self.attention_mask,
            action_mask=self.action_mask,
            rollout_log_probs=self.rollout_log_probs,
            prompts=self.prompts,
            labels=self.labels,
            rewards=self.rewards,
            scores=self.scores,
            info=self.info,
            metadata=[self.metadata] if self.metadata is not None else [],
        )


class RemoteSampleGenerator:
    """Stateless sampler: pulls prompts, dispatches to rollout workers, applies filters."""

    def __init__(
        self,
        strategy,
        tokenizer,
        vllm_engines: List,
        rollout_workers: List,
        dataset_split: str,
        generate_kwargs: dict,
        eval: bool = False,
    ):
        self.strategy = strategy
        self.args = strategy.args
        self.generate_kwargs = generate_kwargs

        self.tokenizer = tokenizer
        self.vllm_engines = vllm_engines or []
        self.rollout_workers = rollout_workers or []

        if not self.vllm_engines:
            raise ValueError("vLLM engines must be provided to build the rollout sampler.")
        if not self.rollout_workers:
            raise ValueError("Rollout workers must be provided to build the rollout sampler.")

        self.prompts_dataloader, self.max_steps = build_prompt_dataloader(
            self.args, strategy, self.tokenizer, dataset_split
        )
        # Track round-robin positions separately in case engines/workers differ.
        self._engine_count = len(self.vllm_engines)
        self._worker_count = len(self.rollout_workers)

    def state_dict(self) -> dict:
        return self.prompts_dataloader.state_dict()

    def load_state_dict(self, state_dict: dict):
        if state_dict:
            self.prompts_dataloader.load_state_dict(state_dict)

    def make_sample_batch(self) -> Tuple[List[Sample], Optional[float], int, bool]:
        """Produce one batch and indicate if the dataloader is exhausted."""
        if getattr(self, "_dataloader_iter", None) is None:
            self._dataloader_iter = iter(self.prompts_dataloader)

        filter_hook = DynamicFilteringHook(self.args) if self.args.dynamic_filtering else NoOpFilterHook()

        # vLLM wakeup when vllm_enable_sleep
        if self.args.vllm_enable_sleep:
            from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call

            batch_vllm_engine_call(self.vllm_engines, "wake_up")

        samples, is_exhausted, prompts_used = self._generate_samples(
            dataloader_iter=self._dataloader_iter,
            num_prompts=self.args.rollout_batch_size,
            filter_hook=filter_hook,
            **self.generate_kwargs,
        )

        # vLLM offload when vllm_enable_sleep
        if self.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "sleep")

        expected_batch = self.args.rollout_batch_size * self.args.n_samples_per_prompt
        if len(samples) < expected_batch:
            # Treat partial batches as exhausted to avoid passing undersized data downstream.
            is_exhausted = True

        pass_rate = filter_hook.pass_rate() if self.args.dynamic_filtering else None
        filter_hook.reset()

        if is_exhausted:
            self._dataloader_iter = None

        return samples, pass_rate, prompts_used, is_exhausted

    @torch.no_grad()
    def _generate_samples(
        self, dataloader_iter, num_prompts: int, **generate_kwargs
    ) -> Tuple[List[Sample], bool, int]:
        """Generate a batch of Samples, applying filter hooks if configured."""
        generate_kwargs = dict(generate_kwargs)  # avoid mutating caller state
        filter_hook: FilterHookBase = generate_kwargs.pop("filter_hook", NoOpFilterHook())

        total_prompt_processed = 0

        prompts, labels, metadatas, exhausted = _collect_prompts(dataloader_iter, num_prompts)
        if exhausted and len(prompts) < num_prompts:
            return [], True, 0

        pending_refs, ref_map = self._dispatch_prompt_requests(prompts, labels, metadatas, **generate_kwargs)
        total_prompt_processed += len(prompts)

        accepted_samples: List[Sample] = []
        target_samples = num_prompts * self.args.n_samples_per_prompt
        pbar = tqdm(range(num_prompts), desc="Generate samples")

        while pending_refs:
            ready_refs, pending_refs = ray.wait(pending_refs, num_returns=1, timeout=10.0)
            for ref in ready_refs:
                info = ref_map.pop(ref, None)
                if info is None:
                    logger.warning("Missing ref info, skipping")
                    continue

                outputs = ray.get(ref)
                for output in outputs:
                    output["metadata"] = info.get("metadata")

                samples = [self._create_sample_from_output(output, **generate_kwargs) for output in outputs]
                kept = filter_hook.apply(samples)
                if kept:
                    accepted_samples.extend(kept)
                    pbar.set_postfix(
                        {
                            "pass_rate": filter_hook.pass_rate(),
                            "prompts_in_batch": total_prompt_processed,
                        }
                    )
                    pbar.update()
                    if len(accepted_samples) >= target_samples:
                        for remaining_ref in pending_refs:
                            ray.cancel(remaining_ref)
                        return accepted_samples[:target_samples], exhausted, total_prompt_processed

                # If nothing kept and still prompts available, top up a new prompt.
                if not kept and not exhausted:
                    new_prompts, new_labels, new_metadatas, exhausted = _collect_prompts(dataloader_iter, 1)
                    if new_prompts:
                        new_refs, new_infos = self._dispatch_prompt_requests(
                            new_prompts, new_labels, new_metadatas, **generate_kwargs
                        )
                        pending_refs.extend(new_refs)
                        ref_map.update(new_infos)
                        total_prompt_processed += len(new_prompts)

        if exhausted and len(accepted_samples) < target_samples:
            return [], True, 0

        return accepted_samples[:target_samples], exhausted, total_prompt_processed

    def _dispatch_prompt_requests(
        self, all_prompts: List[str], all_labels: List[str], all_metadatas: List[Any], **kwargs
    ) -> Tuple[List, Dict]:
        """Dispatch a set of prompts to rollout workers and return refs + metadata map."""
        sampling_params = _build_sampling_params(self.args, **kwargs)
        truncate_length = kwargs.get("prompt_max_len", 1024) + kwargs.get("max_new_tokens", 1024)

        refs = []
        ref_map: Dict = {}
        for idx, (prompt, label, metadata) in enumerate(zip(all_prompts, all_labels, all_metadatas)):
            request_id = f"prompt_{random_uuid()}"
            llm_engine = self.vllm_engines[idx % self._engine_count]
            rollout_worker = self.rollout_workers[idx % self._worker_count]

            ref = rollout_worker.execute.remote(
                llm_engine=llm_engine,
                sampling_params=sampling_params,
                prompt=prompt,
                label=label,
                max_length=truncate_length,
                hf_tokenizer=self.tokenizer,
                request_id=request_id,
                num_samples=self.args.n_samples_per_prompt,
            )
            refs.append(ref)
            ref_map[ref] = {
                "id": request_id,
                "prompt": prompt,
                "label": label,
                "metadata": metadata,
            }

        return refs, ref_map

    def _create_sample_from_output(self, output, **kwargs) -> Sample:
        """Wrap output parsing to keep truncation logic in one place."""
        truncate_length = kwargs.get("prompt_max_len", 1024) + kwargs.get("max_new_tokens", 1024)
        return _build_sample_from_output(output, truncate_length, **kwargs)
