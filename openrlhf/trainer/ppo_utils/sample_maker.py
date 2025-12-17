import heapq
from typing import List, Optional, Tuple

import ray
import torch
from tqdm import tqdm
from vllm import SamplingParams

from openrlhf.datasets import PromptDataset
from openrlhf.datasets.utils import blending_datasets
from openrlhf.trainer.ppo_utils.experience_maker import Experience
from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call
from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)


def _collect_prompts(dataloader_iter, num_prompts: int):
    """Draw up to `num_prompts` items from the prompt dataloader."""
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


class RemoteSampleGenerator:
    """Stateless sampler: pulls prompts and dispatches to rollout workers."""

    def __init__(
        self,
        strategy,
        tokenizer,
        vllm_engines: List,
        prompt_split: str,
        generate_kwargs: dict,
    ):
        self.strategy = strategy
        self.args = strategy.args
        self.generate_kwargs = generate_kwargs

        self.tokenizer = tokenizer
        self.vllm_engines = vllm_engines or []

        self.prompts_dataloader, self.max_steps = self.prepare_datasets(
            prompt_split=prompt_split,
        )

    def prepare_datasets(self, prompt_split):
        args = self.args
        strategy = self.strategy

        # prepare datasets
        train_data = blending_datasets(
            args.prompt_data,
            args.prompt_data_probs,
            strategy,
            args.seed,
            max_count=args.max_samples,
            dataset_split=prompt_split,
        )
        # Create train dataset
        train_data = train_data.select(range(min(args.max_samples, len(train_data))))
        prompts_dataset = PromptDataset(train_data, self.tokenizer, strategy, input_template=args.input_template)
        prompts_dataloader = strategy.setup_dataloader(
            prompts_dataset,
            args.vllm_generate_batch_size,
            True,
            True,
        )

        max_steps = (
            len(prompts_dataset)
            * args.n_samples_per_prompt
            // args.train_batch_size
            * args.num_episodes
            * args.max_epochs
        )
        return prompts_dataloader, max_steps

    def state_dict(self) -> dict:
        return self.prompts_dataloader.state_dict()

    def load_state_dict(self, state_dict: dict):
        if state_dict:
            self.prompts_dataloader.load_state_dict(state_dict)

    def make_sample_batch(self) -> Tuple[List[Experience], Optional[float], int, bool]:
        """Produce one batch and indicate if the dataloader is exhausted."""
        if getattr(self, "_dataloader_iter", None) is None:
            self._dataloader_iter = iter(self.prompts_dataloader)

        # Wake sleeping vLLM engines before dispatching.
        if self.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "wake_up")

        experiences, prompts_used, exhausted = self._generate_samples(
            dataloader_iter=self._dataloader_iter,
            num_prompts=self.args.rollout_batch_size,
            **self.generate_kwargs,
        )

        # Put engines back to sleep when enabled.
        if self.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "sleep")

        pass_rate = (len(experiences) / prompts_used * 100) if prompts_used else None

        if exhausted:
            self._dataloader_iter = None

        return experiences, pass_rate, prompts_used, exhausted

    @torch.no_grad()
    def _generate_samples(
        self, dataloader_iter, num_prompts: int, **generate_kwargs
    ) -> Tuple[List[Experience], bool, int]:
        """Generate a batch of Experiences with optional reward filtering."""
        prompts_used = 0
        prompts, labels, exhausted = _collect_prompts(dataloader_iter, num_prompts)
        # Stop early if the prompt source is fully consumed.
        if exhausted:
            return [], prompts_used, exhausted

        pending_refs = self._dispatch_prompts(prompts, labels, **generate_kwargs)
        prompts_used += len(prompts)

        accepted_experiences: List[Experience] = []
        num_samples = num_prompts * self.args.n_samples_per_prompt
        pbar = tqdm(range(num_prompts), desc="Generate samples")

        while pending_refs:
            ready_refs, pending_refs = ray.wait(pending_refs, num_returns=1, timeout=10.0)
            for ref in ready_refs:
                # Build Experience objects for each output returned from this worker.
                experiences = [
                    self._create_experience_from_output(output, **generate_kwargs) for output in ray.get(ref)
                ]

                # Drop experiences if the average score falls outside the allowed range.
                if self.args.dynamic_filtering and all(e.scores is not None for e in experiences):
                    scores = [e.scores[0].item() for e in experiences]
                    avg_reward = sum(scores) / len(scores)
                    min_r, max_r = self.args.dynamic_filtering_reward_range
                    if not (min_r < avg_reward < max_r):
                        experiences = []

                # Accept experiences and stop once enough have been gathered.
                if experiences:
                    accepted_experiences.extend(experiences)
                    pbar.set_postfix({"prompts_used": prompts_used})
                    pbar.update()
                    # Stop early once the target batch size is reached.
                    if len(accepted_experiences) >= num_samples:
                        for remaining_ref in pending_refs:
                            ray.cancel(remaining_ref)
                        return accepted_experiences[:num_samples], prompts_used, exhausted

                # If rejected, request a new prompt to keep filling the batch.
                else:
                    # Pull another prompt when the current one fails filtering.
                    new_prompts, new_labels, exhausted = _collect_prompts(dataloader_iter, 1)
                    # Cancel outstanding work if the dataloader is drained.
                    if exhausted:
                        for remaining_ref in pending_refs:
                            ray.cancel(remaining_ref)
                        return accepted_experiences[:num_samples], prompts_used, exhausted
                    # Otherwise dispatch the new prompt to keep filling the queue.
                    else:
                        new_refs = self._dispatch_prompts(new_prompts, new_labels, **generate_kwargs)
                        pending_refs.extend(new_refs)
                        prompts_used += len(new_prompts)

        # If the loader is drained and we still lack a full batch, signal exhaustion.
        if exhausted and len(accepted_experiences) < num_samples:
            return [], prompts_used, True

        return accepted_experiences[:num_samples], prompts_used, exhausted

    def _dispatch_prompts(self, prompts: List[str], labels: List[str], **generate_kwargs) -> List:
        """Send prompts to rollout workers and return Ray object refs."""
        sampling_params = SamplingParams(
            temperature=generate_kwargs.get("temperature", 1.0),
            top_p=generate_kwargs.get("top_p", 1.0),
            top_k=generate_kwargs.get("top_k", -1),
            max_tokens=generate_kwargs.get("max_new_tokens", 1024),
            min_tokens=generate_kwargs.get("min_new_tokens", 1),
            skip_special_tokens=generate_kwargs.get("skip_special_tokens", False),
            logprobs=1 if self.args.enable_vllm_is_correction else None,
        )
        truncate_length = generate_kwargs.get("prompt_max_len", 1024) + generate_kwargs.get("max_new_tokens", 1024)

        # Snapshot current unfinished request counts to balance upcoming work.
        unfinished_counts = ray.get([engine.get_num_unfinished_requests.remote() for engine in self.vllm_engines])
        engine_heap = [(count, idx) for idx, count in enumerate(unfinished_counts)]
        heapq.heapify(engine_heap)

        # Pre-compute engine assignment to keep loads even.
        engine_indices = []
        for _ in prompts:
            current_load, engine_idx = heapq.heappop(engine_heap)
            engine_indices.append(engine_idx)
            heapq.heappush(engine_heap, (current_load + self.args.n_samples_per_prompt, engine_idx))

        refs = []
        for idx, (prompt, label) in enumerate(zip(prompts, labels)):
            # Spread work across engines/workers in load-aware order.
            llm_engine = self.vllm_engines[engine_indices[idx]]
            ref = llm_engine.rollout.remote(
                prompt=prompt,
                label=label,
                sampling_params=sampling_params,
                max_length=truncate_length,
                hf_tokenizer=self.tokenizer,
                num_samples=self.args.n_samples_per_prompt,
            )
            refs.append(ref)

        return refs

    def _create_experience_from_output(self, output, **generate_kwargs) -> Experience:
        """Turn a single vLLM response into an Experience."""
        truncate_length = generate_kwargs.get("prompt_max_len", 1024) + generate_kwargs.get("max_new_tokens", 1024)

        # Base rollout fields from the output.
        tokenized_observation = output["observation_tokens"].copy()
        tokenized_ranges = output["action_ranges"]
        reward_val = output.get("reward", None)
        score_val = output.get("scores", None)

        sequences = torch.tensor(tokenized_observation, dtype=torch.long)
        attention_mask = torch.tensor([1] * len(tokenized_observation))
        # Mark the action span within the concatenated tokens.
        action_mask = torch.zeros_like(attention_mask)
        for start, end in tokenized_ranges:
            action_mask[start:end] = 1

        # Truncate everything to the configured context window.
        sequences = sequences[:truncate_length].to("cpu")
        attention_mask = attention_mask[:truncate_length].to("cpu")
        action_mask = action_mask[1:truncate_length].to("cpu")

        # Align rollout logprobs with the truncated action span.
        if output["rollout_log_probs"] is not None:
            rollout_log_probs = torch.tensor(output["rollout_log_probs"][1:truncate_length]).to("cpu")
        else:
            rollout_log_probs = None

        # Collect simple stats about lengths and clipping.
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

        # Convert extra logs to tensors for downstream consumers.
        extra_logs = output.get("extra_logs", {})
        for key, value in extra_logs.items():
            if isinstance(value, torch.Tensor):
                value = value.flatten()[0].item()
            info[key] = torch.tensor([value])

        return Experience(
            sequences=sequences.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
            action_mask=action_mask.unsqueeze(0),
            rollout_log_probs=rollout_log_probs.unsqueeze(0) if rollout_log_probs is not None else None,
            prompts=[output["prompt"]],
            labels=[output["label"]],
            rewards=torch.tensor([reward_val]) if reward_val is not None else None,
            scores=torch.tensor([score_val]) if score_val is not None else None,
            info=info,
        )
