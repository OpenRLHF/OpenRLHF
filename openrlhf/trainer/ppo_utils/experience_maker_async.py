import random
import time
from abc import ABC, abstractmethod
from typing import List, Optional


import ray
import torch
from tqdm import tqdm

from openrlhf.trainer.ppo_utils.experience_maker import Experience
from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)


class FilterHookBase(ABC):
    """Filter hook to optionally drop samples and report pass rate."""

    @abstractmethod
    def apply(self, experiences: List[Experience]) -> List[Experience]:
        raise NotImplementedError

    def pass_rate(self) -> Optional[float]:
        return None

    def reset(self):
        """Reset internal stats if any."""
        return


class NoOpFilterHook(FilterHookBase):
    def apply(self, experiences: List[Experience]) -> List[Experience]:
        return experiences


class DynamicFilteringHook(FilterHookBase):
    """Group-level filtering based on avg reward in a target range."""

    def __init__(self, args):
        self.n_samples = args.n_samples_per_prompt
        self.min_r, self.max_r = args.dynamic_filtering_reward_range
        self.total_groups = 0
        self.valid_groups = 0

    def apply(self, experiences: List[Experience]) -> List[Experience]:
        if len(experiences) != self.n_samples:
            return []
        self.total_groups += 1

        scores = [exp.scores[0].item() for exp in experiences]
        avg_reward = sum(scores) / len(scores)
        
        is_valid = self.min_r < avg_reward < self.max_r
        if is_valid:
            self.valid_groups += 1
            return experiences
        
        logger.info(
            f"Filtered out: avg_reward={avg_reward:.4f}, threshold=({self.min_r:.4f}, {self.max_r:.4f}), scores={scores}"
        )
        return []

    def pass_rate(self) -> Optional[float]:
        if self.total_groups == 0:
            return None
        return self.valid_groups / self.total_groups * 100

    def reset(self):
        self.total_groups = 0
        self.valid_groups = 0


def _collect_prompts(dataloader_iter, num_prompts: int):
    prompts, labels = [], []
    exhausted = False

    while len(prompts) < num_prompts:
        try:
            _, rand_prompts, rand_labels = next(dataloader_iter)
            remaining = num_prompts - len(prompts)
            prompts.extend(rand_prompts[:remaining])
            labels.extend(rand_labels[:remaining])
        except StopIteration:
            exhausted = True
            break

    return prompts, labels, exhausted


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


class SamplesGeneratorAsync:
    def __init__(self, vllm_engines, strategy, tokenizer, prompt_max_len):
        self.strategy = strategy
        self.args = strategy.args
        self.vllm_engines = vllm_engines
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len

    @torch.no_grad()
    def generate_batch(self, dataloader_iter, num_prompts: int, **kwargs) -> tuple[List[Experience], bool, int]:
        filter_hook: FilterHookBase = kwargs.pop("filter_hook", NoOpFilterHook())

        prompts, labels, exhausted = _collect_prompts(dataloader_iter, num_prompts)
        if not prompts:
            return [], True, 0

        # Submit prompts to engines
        refs, _ = self._dispatch_prompt_requests(prompts, labels, **kwargs)
        ray.get(refs)

        # Collect all outputs from engines
        all_output_refs = [llm.get_responses.remote() for llm in self.vllm_engines]
        all_outputs = sum(ray.get(all_output_refs), [])

        # Keep samples from the same prompt grouped
        prompt_groups = {}
        for output in all_outputs:
            prompt_groups.setdefault(output["prompt"], []).append(output)
        all_outputs = [item for group in prompt_groups.values() for item in group]

        # Convert outputs to Experience objects
        experiences_list = []
        for output in all_outputs:
            experiences_list.append(self._create_experience_from_output(output, **kwargs))

        # Apply filter hook in prompt-sized batches
        filtered: List[Experience] = []
        group = self.args.n_samples_per_prompt
        for i in range(0, len(experiences_list), group):
            batch = experiences_list[i : i + group]
            filtered.extend(filter_hook.apply(batch))

        return filtered, exhausted, len(prompts)

    def _dispatch_prompt_requests(self, all_prompts: List[str], all_labels, **kwargs):
        """Send one request per prompt and return refs plus request metadata."""
        llms = self.vllm_engines
        args = self.strategy.args

        sampling_params = _build_sampling_params(args, **kwargs)
        truncate_length = self.prompt_max_len + kwargs.get("max_new_tokens", 1024)

        n_samples_per_prompt = kwargs.get("n_samples_per_prompt", args.n_samples_per_prompt)
        engine_count = len(llms)

        # One request per prompt, round-robin across engines
        refs = []
        infos = []
        for idx, (prompt, label) in enumerate(zip(all_prompts, all_labels)):
            request_id = f"prompt_{time.time()}_{random.randint(1000, 9999)}"
            prompts = [prompt] * n_samples_per_prompt
            labels = [label] * n_samples_per_prompt
            llm = llms[idx % engine_count]
            ref = llm.add_requests.remote(
                sampling_params=sampling_params,
                prompts=prompts,
                labels=labels,
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
                }
            )

        return refs, infos

    def _create_experience_from_output(self, output, **kwargs):
        """Build an Experience from a single vLLM output."""
        # Use already tokenized observation
        tokenized_observation = output["observation_tokens"].copy()
        # Action ranges are token indices
        tokenized_ranges = output["action_ranges"]

        # Build tensors
        sequences = torch.tensor(tokenized_observation, dtype=torch.long)
        attention_mask = torch.tensor([1] * len(tokenized_observation))

        # Build action mask from token ranges
        action_mask = torch.zeros_like(attention_mask)
        # Mark action positions in the mask
        for start, end in tokenized_ranges:
            action_mask[start:end] = 1

        # Apply length limit
        truncate_length = self.prompt_max_len + kwargs.get("max_new_tokens", 1024)
        sequences = sequences[:truncate_length].to("cpu")
        attention_mask = attention_mask[:truncate_length].to("cpu")
        action_mask = action_mask[1:truncate_length].to("cpu")
        if output["rollout_log_probs"] is not None:
            rollout_log_probs = torch.tensor(output["rollout_log_probs"][1:truncate_length]).to("cpu")
        else:
            rollout_log_probs = None

        # Compute response length (first to last action token)
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

        # Process extra_logs
        extra_logs = output.get("extra_logs", {})
        for key, value in extra_logs.items():
            info[key] = torch.tensor([value.item()])

        experience = Experience(
            sequences=sequences.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
            action_mask=action_mask.unsqueeze(0),
            rollout_log_probs=rollout_log_probs.unsqueeze(0) if rollout_log_probs is not None else None,
            prompts=[output["prompt"]],
            labels=[output["label"]],
            rewards=torch.tensor([output["reward"]]),
            scores=torch.tensor([output["scores"]]),
            info=info,
        )

        return experience


class SamplesGeneratorStreamingAsync(SamplesGeneratorAsync):

    @torch.no_grad()
    def generate_batch(self, dataloader_iter, num_prompts: int, **kwargs) -> tuple[List[Experience], bool, int]:
        """Stream samples until target prompts are collected or the dataloader is exhausted."""
        filter_hook: FilterHookBase = kwargs.pop("filter_hook", NoOpFilterHook())

        total_prompt_processed = 0

        # Collect initial prompts
        prompts, labels, exhausted = _collect_prompts(dataloader_iter, num_prompts)
        if not prompts:
            return [], True, 0

        # Submit prompts to engines
        remaining_refs, remaining_infos = self._dispatch_prompt_requests(prompts, labels, **kwargs)
        ref_map = {info["ref"]: info for info in remaining_infos}
        total_prompt_processed += len(prompts)

        valid_experiences = []
        num_samples = num_prompts * self.args.n_samples_per_prompt
        pbar = tqdm(
            range(num_prompts),
            desc=f"Generate experiences",
            disable=False,
        )
        while remaining_refs:
            # Wait for the next completed request
            ready_refs, remaining_refs = ray.wait(remaining_refs, num_returns=1, timeout=10.0)
            for ref in ready_refs:
                info = ref_map.pop(ref, None)
                if info is None:
                    logger.warning("Missing ref info, skipping")
                    continue

                # Collect output from engines
                outputs = ray.get(info["llm"].get_responses.remote(info["id"]))
                # Convert output to Experience objects
                experiences_list = [self._create_experience_from_output(output, **kwargs) for output in outputs]

                # Filter kept experiences
                kept = filter_hook.apply(experiences_list)
                if kept:
                    valid_experiences.extend(kept)
                    pbar.update()

                    if len(valid_experiences) >= num_samples:
                        # Cancel remaining requests
                        for ref in remaining_refs:
                            ray.cancel(ref)

                        return valid_experiences[:num_samples], exhausted, total_prompt_processed
                elif not exhausted:
                    # Top up requests when a prompt is filtered out
                    prompts, labels, exhausted = _collect_prompts(dataloader_iter, 1)
                    if not prompts:
                        continue

                    new_refs, new_infos = self._dispatch_prompt_requests(prompts, labels, **kwargs)
                    if new_refs is not None:
                        remaining_refs.extend(new_refs)
                        ref_map.update({info["ref"]: info for info in new_infos})
                        total_prompt_processed += len(prompts)

        return (
            valid_experiences[:num_samples],
            exhausted,
            total_prompt_processed,
        )
