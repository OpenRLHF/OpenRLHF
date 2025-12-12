import random
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import ray
import torch

from openrlhf.trainer.ppo_utils.experience_maker import Experience
from openrlhf.trainer.ppo_utils.filter_hooks import DynamicFilteringHook, FilterHookBase, NoOpFilterHook
from openrlhf.trainer.ppo_utils.misc import build_prompt_dataloader
from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)


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
        "reward": torch.tensor([output["reward"]]),
        "score": torch.tensor([output["scores"]]),
    }

    extra_logs = output.get("extra_logs", {})
    for key, value in extra_logs.items():
        info[key] = torch.tensor([value.item()])

    return Sample(
        sequences=sequences.unsqueeze(0),
        attention_mask=attention_mask.unsqueeze(0),
        action_mask=action_mask.unsqueeze(0),
        rollout_log_probs=rollout_log_probs.unsqueeze(0) if rollout_log_probs is not None else None,
        prompts=[output["prompt"]],
        labels=[output["label"]],
        rewards=torch.tensor([output["reward"]]),
        scores=torch.tensor([output["scores"]]),
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


class RemoteSampleStreamer:
    """Stateless sampler: caller decides which split to build; no train/eval knowledge here."""

    def __init__(
        self,
        strategy,
        tokenizer,
        vllm_engines: List,
        prompt_max_len: int,
        dataset_split: str,
        generate_kwargs: dict,
        for_eval: bool = False,
    ):
        self.strategy = strategy
        self.args = strategy.args
        self.prompt_max_len = prompt_max_len
        self.generate_kwargs = generate_kwargs

        self.tokenizer = tokenizer
        self.vllm_engines = vllm_engines

        self.prompts_dataloader, self.max_steps = build_prompt_dataloader(
            self.args, strategy, self.tokenizer, dataset_split, for_eval=for_eval
        )

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
        samples, is_exhausted, prompts_used = self._generate_batch(
            dataloader_iter=self._dataloader_iter,
            num_prompts=self.args.rollout_batch_size,
            filter_hook=filter_hook,
            **self.generate_kwargs,
        )
        expected_batch = self.args.rollout_batch_size * self.args.n_samples_per_prompt
        if len(samples) < expected_batch:
            # Treat partial batches as exhausted to avoid passing undersized data downstream.
            is_exhausted = True

        pass_rate = filter_hook.pass_rate() if self.args.dynamic_filtering else None
        filter_hook.reset()

        if is_exhausted:
            self._dataloader_iter = None

        return samples, pass_rate, prompts_used, is_exhausted

    # def iter_eval_prompts(self) -> Iterator[Tuple[List[str], List[str], List[str]]]:
    #     """Yield eval datasource/prompt/label triples through the streamer interface."""
    #     if not self.prompts_dataloader:
    #         return
    #     for datasources, prompts, labels in self.prompts_dataloader:
    #         yield datasources, prompts, labels

    # @torch.no_grad()
    # def generate_samples(self, all_prompts: List[str], all_labels: List[str], **kwargs) -> List[Sample]:
    #     """Generate samples for a provided prompt list (used in eval)."""
    #     refs, infos = self._dispatch_prompt_requests(all_prompts, all_labels, **kwargs)
    #     ref_map = {info["ref"]: info for info in infos}
    #     total_needed = len(all_prompts) * kwargs.get("n_samples_per_prompt", self.args.n_samples_per_prompt)

    #     collected = []
    #     while refs and len(collected) < total_needed:
    #         ready_refs, refs = ray.wait(refs, num_returns=1)
    #         for ref in ready_refs:
    #             info = ref_map.pop(ref, None)
    #             if info is None:
    #                 continue
    #             outputs = ray.get(info["llm"].get_responses.remote(info["id"]))
    #             collected.extend([self._create_sample_from_output(output, **kwargs) for output in outputs])

    #     return collected[:total_needed]

    @torch.no_grad()
    def _generate_batch(self, dataloader_iter, num_prompts: int, **kwargs) -> Tuple[List[Sample], bool, int]:
        """Generate a batch of Samples, applying filter hooks if configured."""
        filter_hook: FilterHookBase = kwargs.pop("filter_hook", NoOpFilterHook())

        total_prompt_processed = 0

        prompts, labels, metadatas, exhausted = _collect_prompts(dataloader_iter, num_prompts)
        if exhausted and len(prompts) < num_prompts:
            return [], True, 0

        remaining_refs, remaining_infos = self._dispatch_prompt_requests(prompts, labels, metadatas, **kwargs)
        ref_map = {info["ref"]: info for info in remaining_infos}
        total_prompt_processed += len(prompts)

        accepted_samples = []
        num_samples = num_prompts * self.args.n_samples_per_prompt
        while remaining_refs:
            ready_refs, remaining_refs = ray.wait(remaining_refs, num_returns=1, timeout=10.0)
            for ref in ready_refs:
                info = ref_map.pop(ref, None)
                if info is None:
                    logger.warning("Missing ref info, skipping")
                    continue

                outputs = ray.get(info["llm"].get_responses.remote(info["id"]))
                for output in outputs:
                    output["metadata"] = info.get("metadata")
                sample_list = [self._create_sample_from_output(output, **kwargs) for output in outputs]

                kept = filter_hook.apply(sample_list)
                if kept:
                    accepted_samples.extend(kept)
                    if len(accepted_samples) >= num_samples:
                        for ref in remaining_refs:
                            ray.cancel(ref)
                        return accepted_samples[:num_samples], exhausted, total_prompt_processed
                elif not exhausted:
                    prompts, labels, metadatas, exhausted = _collect_prompts(dataloader_iter, 1)
                    if not prompts:
                        continue

                    new_refs, new_infos = self._dispatch_prompt_requests(prompts, labels, metadatas, **kwargs)
                    if new_refs is not None:
                        remaining_refs.extend(new_refs)
                        ref_map.update({info["ref"]: info for info in new_infos})
                        total_prompt_processed += len(prompts)

        if exhausted and len(accepted_samples) < num_samples:
            return [], True, 0

        return accepted_samples[:num_samples], exhausted, total_prompt_processed

    def _dispatch_prompt_requests(
        self, all_prompts: List[str], all_labels: List[str], all_metadatas: List[Any], **kwargs
    ) -> Tuple[List, List[dict]]:
        llms = self.vllm_engines
        args = self.strategy.args

        sampling_params = _build_sampling_params(args, **kwargs)
        truncate_length = self.prompt_max_len + kwargs.get("max_new_tokens", 1024)

        n_samples_per_prompt = kwargs.get("n_samples_per_prompt", args.n_samples_per_prompt)
        engine_count = len(llms)

        refs = []
        infos = []
        for idx, (prompt, label, metadata) in enumerate(zip(all_prompts, all_labels, all_metadatas)):
            request_id = f"prompt_{time.time()}_{random.randint(1000, 9999)}"
            batched_prompts = [prompt] * n_samples_per_prompt
            batched_labels = [label] * n_samples_per_prompt
            llm = llms[idx % engine_count]
            ref = llm.add_requests.remote(
                sampling_params=sampling_params,
                prompts=batched_prompts,
                labels=batched_labels,
                max_length=truncate_length,
                hf_tokenizer=self.tokenizer,
                request_group_id=request_id,
            )
            refs.append(ref)
            infos.append(
                {
                    "id": request_id,
                    "llm": llm,
                    "ref": ref,
                    "prompt": prompt,
                    "label": label,
                    "metadata": metadata,
                }
            )

        return refs, infos

    def _create_sample_from_output(self, output, **kwargs) -> Sample:
        """Wrap output parsing to keep truncation logic in one place."""
        truncate_length = self.prompt_max_len + kwargs.get("max_new_tokens", 1024)
        return _build_sample_from_output(output, truncate_length, **kwargs)
