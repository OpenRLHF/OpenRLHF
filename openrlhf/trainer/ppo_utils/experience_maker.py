import heapq
import time
from copy import deepcopy
from dataclasses import dataclass, fields
from datetime import timedelta
from typing import Any, List, Optional, Tuple, Union

import ray
import torch
from tqdm import tqdm
from vllm import SamplingParams

from openrlhf.models.utils import compute_approx_kl, compute_reward, masked_mean
from openrlhf.trainer.ppo_utils.length_penalty import apply_length_penalties
from openrlhf.trainer.ray.launcher import RayActorGroup
from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.seqlen_balancing import get_minimum_num_micro_batch_size, get_seqlen_balanced_partitions
from openrlhf.utils.utils import zero_pad_sequences

logger = init_logger(__name__)


def to(tensor: Union[torch.Tensor, list[torch.Tensor]], device):
    if isinstance(tensor, list):
        return [to(t, device) for t in tensor]
    return tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor


def pin_memory(tensor: Union[torch.Tensor, list[torch.Tensor]]):
    if isinstance(tensor, list):
        return [pin_memory(t) for t in tensor]
    return tensor.pin_memory() if isinstance(tensor, torch.Tensor) else tensor


@dataclass
class Experience:
    """Experience is a batch of data for RLHF training.

    Shapes of each tensor:
    index: (B,)
    sequences: (B, S)
    attention_mask: (B, S)
    action_mask: (B, A)
    action_log_probs: (B, S)
    base_action_log_probs: (B, S)
    values: (B, S)
    returns: (B, S)
    advantages: (B, S)
    kl: (B, S)
    info: dict[str, list]
    """

    index: list[int] = None
    sequences: torch.Tensor = None
    attention_mask: torch.LongTensor = None
    action_mask: torch.BoolTensor = None

    action_log_probs: torch.Tensor = None
    base_action_log_probs: torch.Tensor = None
    rollout_log_probs: torch.Tensor = None
    values: torch.Tensor = None
    returns: torch.Tensor = None
    advantages: torch.Tensor = None
    kl: torch.Tensor = None

    prompts: list[str] = None
    labels: list[str] = None
    rewards: torch.Tensor = None  # used for advantage calculation
    scores: torch.Tensor = None  # 0-1 reward used for dynamic sampling

    # the info field is used to store additional information
    # all the fields in the info will be logged to the tensorboard/wandb
    info: dict[str, torch.Tensor] = None

    def __init__(
        self,
        index=None,
        sequences=None,
        action_log_probs=None,
        base_action_log_probs=None,
        rollout_log_probs=None,
        values=None,
        returns=None,
        advantages=None,
        attention_mask=None,
        action_mask=None,
        kl=None,
        prompts=None,
        labels=None,
        rewards=None,
        scores=None,
        info=None,
    ):
        self.index = index
        self.sequences = sequences
        self.action_log_probs = action_log_probs
        self.base_action_log_probs = base_action_log_probs
        self.rollout_log_probs = rollout_log_probs
        self.values = values
        self.returns = returns
        self.advantages = advantages
        self.attention_mask = attention_mask
        self.action_mask = action_mask
        self.kl = kl
        self.prompts = prompts or []
        self.labels = labels or []
        self.rewards = rewards
        self.scores = scores
        self.info = info or []

    @torch.no_grad()
    def to_device(self, device: torch.device):
        """Move all tensor fields to the specified device."""
        for field, value in self.__dict__.items():
            if isinstance(value, dict):
                setattr(self, field, {key: to(val, device) for key, val in value.items()})
            else:
                setattr(self, field, to(value, device))

        return self

    def pin_memory(self):
        """Pin memory for all tensor fields."""
        for field, value in self.__dict__.items():
            if isinstance(value, dict):
                setattr(self, field, {key: pin_memory(val) for key, val in value.items()})
            else:
                setattr(self, field, pin_memory(value))

        return self

    @staticmethod
    def select(experiences: List["Experience"], fields: List[str]) -> List["Experience"]:
        """Select specific fields from a list of Experience instances to create new Experience instances.

        Args:
            experiences: List of Experience instances
            fields: List of field names to select

        Returns:
            A list of new Experience instances containing only the selected fields
        """
        new_experiences = []
        for exp in experiences:
            new_exp = Experience()
            for field in fields:
                if hasattr(exp, field):
                    setattr(new_exp, field, getattr(exp, field))
            new_experiences.append(new_exp)
        return new_experiences

    @staticmethod
    def _merge_item(items: List, pad_value: int = 0) -> Union[torch.Tensor, list, dict, Any]:
        """Merge a list of items into a single item.
        Recursively merge tensors, lists and dicts.
        For tensors, use zero_pad_sequences to merge sequences of different lengths.

        Args:
            items: List of items to merge
            pad_value: Value used for padding tensors
        """
        if isinstance(items[0], torch.Tensor):
            return zero_pad_sequences(items, side="right", value=pad_value)
        elif isinstance(items[0], list):
            return sum(items, [])
        elif isinstance(items[0], dict):
            result = {}
            # Collect all values for each key
            for d in items:
                for key, value in d.items():
                    if key not in result:
                        result[key] = []
                    result[key].append(value)
            # Merge all values for each key at once
            return {key: Experience._merge_item(values, pad_value) for key, values in result.items()}
        elif items[0] is None:
            return None
        else:
            raise ValueError(f"Unsupported type: {type(items[0])}")

    @staticmethod
    def concat_experiences(experiences_list: List["Experience"], pad_token_id) -> "Experience":
        """Concatenate multiple experiences into one large experience.

        Args:
            experiences_list: List of Experience to concatenate
            pad_token_id: Token id used for padding sequences

        Returns:
            A new Experience instance containing all the concatenated data
        """
        if not experiences_list:
            return Experience()

        # Get all field names from the dataclass
        field_names = [f.name for f in fields(Experience)]

        # Create result dictionary
        result = {}

        # Merge all fields
        for field in field_names:
            values = [getattr(e, field) for e in experiences_list]
            # Use pad_token_id for sequences field, 0 for others
            pad_value = pad_token_id if field == "sequences" else 0
            result[field] = Experience._merge_item(values, pad_value)

        return Experience(**result)


def _collect_prompt_batch(dataloader_iter, num_prompts: int):
    """Draw up to `num_prompts` items from the prompt dataloader."""
    prompts, labels = [], []
    exhausted = False

    while len(prompts) < num_prompts:
        try:
            _, batch_prompts, batch_labels = next(dataloader_iter)
            remaining = num_prompts - len(prompts)
            prompts.extend(batch_prompts[:remaining])
            labels.extend(batch_labels[:remaining])
        except StopIteration:
            exhausted = True
            break

    return prompts, labels, exhausted


class SamplesGenerator:
    """Stateless sample generator: pulls prompts and dispatches to rollout workers."""

    def __init__(
        self,
        strategy,
        prompts_dataloader,
        eval_dataloader,
        tokenizer,
        vllm_engines: List,
    ):
        self.strategy = strategy
        self.args = strategy.args

        self.tokenizer = tokenizer
        self.vllm_engines = vllm_engines or []

        self.prompts_dataloader = prompts_dataloader
        self.eval_dataloader = eval_dataloader

    @torch.no_grad()
    def generate_eval_samples(self, **generate_kwargs) -> Tuple[List[Experience], Optional[float], int, bool]:
        if getattr(self, "_eval_dataloader_iter", None) is None:
            self._eval_dataloader_iter = iter(self.eval_dataloader)

        # Wake sleeping vLLM engines before dispatching.
        if self.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "wake_up")

        experiences, prompts_consumed, exhausted = self._generate_vllm(
            dataloader_iter=self._eval_dataloader_iter,
            num_prompts=len(self.eval_dataloader),
            dynamic_filtering=False,
            **generate_kwargs,
        )

        # Put engines back to sleep when enabled.
        if self.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "sleep")

        self._eval_dataloader_iter = None

        return experiences

    @torch.no_grad()
    def generate_samples(self, **generate_kwargs) -> Tuple[List[Experience], Optional[float], int, bool]:
        """Produce one batch and indicate if the dataloader is exhausted."""
        if getattr(self, "_dataloader_iter", None) is None:
            self._dataloader_iter = iter(self.prompts_dataloader)

        # Wake sleeping vLLM engines before dispatching.
        if self.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "wake_up")

        experiences, prompts_consumed, exhausted = self._generate_vllm(
            dataloader_iter=self._dataloader_iter,
            num_prompts=self.args.rollout_batch_size,
            dynamic_filtering=self.args.dynamic_filtering,
            **generate_kwargs,
        )

        # Put engines back to sleep when enabled.
        if self.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "sleep")

        filter_pass_rate = None
        if self.args.dynamic_filtering and prompts_consumed:
            filter_pass_rate = self.args.rollout_batch_size / prompts_consumed * 100

        if exhausted:
            self._dataloader_iter = None
            logger.info("Prompt dataloader is exhausted.")

        return experiences, filter_pass_rate, prompts_consumed, exhausted

    def _generate_vllm(
        self, dataloader_iter, num_prompts: int, dynamic_filtering, **generate_kwargs
    ) -> Tuple[List[Experience], int, bool]:
        """Generate a batch of Experiences with optional reward filtering."""
        prompts_consumed = 0
        prompts, labels, exhausted = _collect_prompt_batch(dataloader_iter, num_prompts)
        # Stop early if the prompt source is fully consumed.
        if exhausted:
            return [], prompts_consumed, exhausted

        pending_refs = self._dispatch_prompts_to_vllm(prompts, labels, **generate_kwargs)
        prompts_consumed += len(prompts)

        accepted_experiences: List[Experience] = []
        pbar = tqdm(range(num_prompts), desc="Generate samples")

        while pending_refs:
            ready_refs, pending_refs = ray.wait(pending_refs, num_returns=1, timeout=10.0)
            for ref in ready_refs:
                # Build Experience objects for each vLLM response returned from this worker.
                experiences = [
                    self._process_response_into_experience(response, **generate_kwargs) for response in ray.get(ref)
                ]

                # Drop experiences if the average score falls outside the allowed range.
                if dynamic_filtering and all(e.scores is not None for e in experiences):
                    scores = [e.scores[0].item() for e in experiences]
                    avg_reward = sum(scores) / len(scores)
                    min_r, max_r = self.args.dynamic_filtering_reward_range
                    if not (min_r < avg_reward < max_r):
                        logger.info(
                            f"Filtered out: avg_reward={avg_reward:.2f}, threshold=({min_r:.2f}, {max_r:.2f}), scores={[f'{s:.2f}' for s in scores]}"
                        )
                        experiences = []

                # Accept experiences and stop once enough have been gathered.
                if experiences:
                    accepted_experiences.extend(experiences)
                    pbar.set_postfix({"prompts_consumed": prompts_consumed})
                    pbar.update()

                # If rejected, request a new prompt to keep filling the batch.
                else:
                    # Pull another prompt when the current one fails filtering.
                    new_prompts, new_labels, exhausted = _collect_prompt_batch(dataloader_iter, 1)
                    prompts_consumed += len(new_prompts)
                    # Cancel outstanding work if the dataloader is drained.
                    if exhausted:
                        for remaining_ref in pending_refs:
                            ray.cancel(remaining_ref)
                        return [], prompts_consumed, True
                    # Otherwise dispatch the new prompt to keep filling the queue.
                    else:
                        new_refs = self._dispatch_prompts_to_vllm(new_prompts, new_labels, **generate_kwargs)
                        pending_refs.extend(new_refs)

        return accepted_experiences, prompts_consumed, exhausted

    def _dispatch_prompts_to_vllm(self, prompts: List[str], labels: List[str], **generate_kwargs) -> List:
        """Send prompts to rollout executors and return Ray object refs."""
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

        # Snapshot current pending rollout counts to balance upcoming work.
        pending_counts = ray.get([engine.get_num_unfinished_requests.remote() for engine in self.vllm_engines])
        engine_heap = [(count, idx) for idx, count in enumerate(pending_counts)]
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
            ref = llm_engine.generate_responses.remote(
                prompt=prompt,
                label=label,
                sampling_params=sampling_params,
                max_length=truncate_length,
                hf_tokenizer=self.tokenizer,
                num_samples=self.args.n_samples_per_prompt,
            )
            refs.append(ref)

        return refs

    def _process_response_into_experience(self, response, **generate_kwargs) -> Experience:
        """Turn a single vLLM response into an Experience."""
        truncate_length = generate_kwargs.get("prompt_max_len", 1024) + generate_kwargs.get("max_new_tokens", 1024)

        # Base rollout fields from the output.
        tokenized_observation = response["observation_tokens"].copy()
        tokenized_ranges = response["action_ranges"]
        reward_val = response.get("reward", None)
        score_val = response.get("scores", None)

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
        if response["rollout_log_probs"] is not None:
            rollout_log_probs = torch.tensor(response["rollout_log_probs"][1:truncate_length]).to("cpu")
        else:
            rollout_log_probs = None

        # Collect simple stats about lengths and clipping.
        ones_indices = torch.where(action_mask)[0]
        response_length = (ones_indices[-1] - ones_indices[0] + 1).item() if len(ones_indices) else 0
        total_length = attention_mask.float().sum()
        is_clipped = total_length >= truncate_length

        # Check if response was truncated (hit max_tokens limit, finish_reason == "length")
        is_truncated = response.get("truncated", False)

        info = {
            "response_length": torch.tensor([response_length]),
            "total_length": torch.tensor([total_length]),
            "response_clip_ratio": torch.tensor([is_clipped]),
            "truncated": torch.tensor([is_truncated]),
        }
        if reward_val is not None:
            info["reward"] = torch.tensor([reward_val])
        if score_val is not None:
            info["score"] = torch.tensor([score_val])

        # Convert extra logs to tensors for downstream consumers.
        extra_logs = response.get("extra_logs", {})
        for key, value in extra_logs.items():
            if isinstance(value, torch.Tensor):
                value = value.flatten()[0].item()
            info[key] = torch.tensor([value])

        return Experience(
            sequences=sequences.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
            action_mask=action_mask.unsqueeze(0),
            rollout_log_probs=rollout_log_probs.unsqueeze(0) if rollout_log_probs is not None else None,
            prompts=[response["prompt"]],
            labels=[response["label"]],
            rewards=torch.tensor([reward_val]) if reward_val is not None else None,
            scores=torch.tensor([score_val]) if score_val is not None else None,
            info=info,
        )


class RemoteExperienceMaker:
    def __init__(
        self,
        actor_model_group: RayActorGroup,
        critic_model_group: RayActorGroup,
        reward_model_group: RayActorGroup,
        initial_model_group: RayActorGroup,
        kl_controller,
        strategy,
        tokenizer,
        **kwargs,
    ):
        super().__init__()

        self.strategy = strategy
        self.args = strategy.args
        self.advantage_estimator = strategy.args.advantage_estimator

        self.actor_model_group = actor_model_group
        self.critic_model_group = critic_model_group
        self.reward_model_group = reward_model_group
        self.initial_model_group = initial_model_group
        self.tokenizer = tokenizer
        self.kl_ctl = kl_controller

    def split_rollout_samples(self, rollout_samples):
        for i, sample in enumerate(rollout_samples):
            sample.index = [i]

        samples_list = []
        if self.args.use_dynamic_batch:
            total_lengths = [int(s.info["total_length"].item()) for s in rollout_samples]
            effective_actor_num = (
                self.args.actor_num_nodes
                * self.args.actor_num_gpus_per_node
                // self.args.ring_attn_size
                // self.args.ds_tensor_parallel_size
            )
            minimum_batch_num = get_minimum_num_micro_batch_size(
                total_lengths,
                self.args.rollout_max_tokens_per_gpu,
                self.args.ring_attn_size,
                self.args.ds_tensor_parallel_size,
            )
            minimum_batch_num = minimum_batch_num // effective_actor_num * effective_actor_num
            num_batch = max(minimum_batch_num, effective_actor_num)
            batch_indexes = get_seqlen_balanced_partitions(total_lengths, num_batch, False)
            for micro_index in batch_indexes:
                micro_batch = [rollout_samples[idx] for idx in micro_index]
                concat_samples = Experience.concat_experiences(micro_batch, self.tokenizer.pad_token_id)
                samples_list.append(concat_samples)
        else:
            batch_size = self.args.micro_rollout_batch_size
            for i in range(0, len(rollout_samples), batch_size):
                concat_samples = Experience.concat_experiences(
                    rollout_samples[i : i + batch_size], self.tokenizer.pad_token_id
                )
                samples_list.append(concat_samples)
        return samples_list

    @torch.no_grad()
    def make_experience_batch(self, rollout_samples) -> List[Experience]:
        """
        Make a list of experience with the micro_rollout_batch_size.

        This method will first calculate the response sequences and rewards for the given prompts.
        Then, if we need certain processing for the rewards or do certain filtering, we can process the rollout as a whole.
        After that, we will calculate the advantages and returns for each experience.
        """
        # Each batch of samples will be scheduled to a effective Ray Actor (i.e, a DP rank)
        samples_list = self.split_rollout_samples(rollout_samples)

        # Make experiences (models forward: logprobs, values, rewards, and kl divergence)
        experiences = self.make_experience(samples_list)

        # Process experiences (reward shaping, etc.)
        experiences = self.compute_advantages_and_returns(experiences)
        return experiences

    @torch.no_grad()
    def make_experience(self, samples_list: List[Experience]) -> List[Experience]:
        """
        Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
        """
        start_time = time.time()
        logger.info(f"🚀 Starting experience making with {sum([len(s.sequences) for s in samples_list])} samples")

        args = self.strategy.args
        device = "cpu"

        # Extract all information from samples in one pass
        # Convert samples into lists of tensors and metadata for batch processing
        sequences_list = [s.sequences for s in samples_list]
        attention_mask_list = [s.attention_mask for s in samples_list]
        action_mask_list = [s.action_mask for s in samples_list]

        # The rewards are already filled in the samples_list, such as the agent's environment rewards
        use_reward_model = samples_list[0].rewards is None
        if use_reward_model:
            if self.reward_model_group is None:
                raise ValueError("reward_model_group is required when rewards are not precomputed")
            # Batch call reward model
            r_refs = self.reward_model_group.async_run_method_batch(
                method_name="forward",
                sequences=sequences_list,
                attention_mask=attention_mask_list,
                pad_sequence=[True] * len(samples_list),
            )
        else:
            r_refs = None

        # Sync to avoid GPU OOM when colocate models
        if args.colocate_all_models and r_refs is not None:
            ray.get(r_refs)
            ray.get(self.reward_model_group.async_run_method(method_name="empty_cache"))

        # Batch call actor model
        action_log_probs_ref = self.actor_model_group.async_run_method_batch(
            method_name="forward",
            sequences=sequences_list,
            action_mask=action_mask_list,
            attention_mask=attention_mask_list,
        )

        # Sync to avoid GPU OOM when colocate models
        if args.colocate_all_models or args.colocate_actor_ref:
            ray.get(action_log_probs_ref)
            ray.get(self.actor_model_group.async_run_method(method_name="empty_cache"))

        # Batch call critic model
        if self.critic_model_group is not None:
            if args.colocate_critic_reward and r_refs is not None:
                ray.get(r_refs)
                ray.get(self.reward_model_group.async_run_method(method_name="empty_cache"))

            value_ref = self.critic_model_group.async_run_method_batch(
                method_name="forward",
                sequences=sequences_list,
                action_mask=action_mask_list,
                attention_mask=attention_mask_list,
            )
            if args.colocate_all_models or args.colocate_critic_reward:
                ray.get(value_ref)
                ray.get(self.critic_model_group.async_run_method(method_name="empty_cache"))
        else:
            value_ref = ray.put([[None]] * (len(samples_list) * args.ring_attn_size * args.ds_tensor_parallel_size))

        # Batch call initial model
        if self.initial_model_group is not None:
            base_action_log_probs_ref = self.initial_model_group.async_run_method_batch(
                method_name="forward",
                sequences=sequences_list,
                action_mask=action_mask_list,
                attention_mask=attention_mask_list,
            )

            if args.colocate_all_models or args.colocate_actor_ref:
                ray.get(base_action_log_probs_ref)
                ray.get(self.initial_model_group.async_run_method(method_name="empty_cache"))
        else:
            base_action_log_probs_ref = ray.put(
                [[None]] * (len(samples_list) * args.ring_attn_size * args.ds_tensor_parallel_size)
            )

        # Wait for all remote calls to complete and flatten the results
        # Note: the results duplicated ring_attn_size * ds_tensor_parallel_size times
        # This is because the actors in ring group and tp group will return the same output
        duplicate_factor = args.ring_attn_size * args.ds_tensor_parallel_size
        action_log_probs_list = sum(ray.get(action_log_probs_ref)[::duplicate_factor], [])
        base_action_log_probs_list = sum(ray.get(base_action_log_probs_ref)[::duplicate_factor], [])
        value_list = sum(ray.get(value_ref)[::duplicate_factor], [])

        # Process rewards based on source
        if use_reward_model:
            # Reward Model
            rewards_list = sum(ray.get(r_refs)[::duplicate_factor], [])
            for i, samples in enumerate(samples_list):
                samples.rewards = rewards_list[i]
                samples.info["reward"] = rewards_list[i]

        assert (
            len(samples_list) == len(action_log_probs_list) == len(base_action_log_probs_list) == len(value_list)
        ), f"len(samples_list): {len(samples_list)}, len(action_log_probs_list): {len(action_log_probs_list)}, len(base_action_log_probs_list): {len(base_action_log_probs_list)}, len(value_list): {len(value_list)}"

        # Process results for each sample
        for i, (samples, action_log_probs, base_action_log_probs, value) in enumerate(
            zip(samples_list, action_log_probs_list, base_action_log_probs_list, value_list)
        ):
            if (self.initial_model_group is not None) and (not args.use_kl_loss):
                kl = compute_approx_kl(
                    action_log_probs,
                    base_action_log_probs,
                    kl_estimator=self.strategy.args.kl_estimator,
                )
                logprobs_diff = action_log_probs.float() - base_action_log_probs.float()
            else:
                kl = torch.zeros_like(action_log_probs, dtype=action_log_probs.dtype, device=device)
                logprobs_diff = torch.zeros_like(action_log_probs, dtype=action_log_probs.dtype, device=device)
            kl_mean = masked_mean(kl, samples.action_mask, dim=-1)
            logprobs_diff_mean = masked_mean(logprobs_diff, samples.action_mask, dim=-1)

            if not args.use_kl_loss:
                base_action_log_probs = None

            # Update experience with new information
            samples.action_log_probs = action_log_probs
            samples.base_action_log_probs = base_action_log_probs
            samples.values = value
            samples.kl = kl
            samples.info["kl"] = kl_mean
            samples.info["logprobs_diff"] = logprobs_diff_mean

        end_time = time.time()
        duration = end_time - start_time
        time_str = str(timedelta(seconds=duration)).split(".")[0]
        logger.info(f"✨ Experience making completed in {time_str}")
        return samples_list

    @torch.no_grad()
    def compute_advantages_and_returns(
        self, experiences: List[Experience], **kwargs
    ) -> Tuple[List[Experience], List[torch.Tensor]]:
        """
        Process experiences, this can be used to filter out some experiences or do some processing on the rewards.
        Example, use_dynamic_batch
            >>> rewards: [0, 1, 0.5, 1], indices: [1, 2, 0, 3], n_samples_per_prompt: 2
            >>> sorted rewards: [0,5, 0, 1, 1], reward shaping: [0.25, 0.25, 1, 1]
            >>> map back: [0.25, 1, 0.25, 1]
        Output:
        - experiences: List of Experience
        - rewards: List of rewards
        """
        args = self.strategy.args

        # Apply length penalties (DAPO overlong / ProRL stop properly) - BEFORE dynamic indices processing
        apply_length_penalties(experiences, args)

        # get rewards from experiences
        exp_len = [len(experience.index) for experience in experiences]
        # indices is an identity mapping when not using dynamic batch; otherwise, it maps back to the original indices after rearrange samples
        indices = torch.tensor(sum([experience.index for experience in experiences], []))
        raw_rewards = torch.cat([experience.rewards for experience in experiences], dim=0)
        rewards = torch.empty_like(raw_rewards)
        rewards[indices] = raw_rewards  # sorted

        rewards = rewards.reshape(-1, args.n_samples_per_prompt)

        # log group reward std
        if args.n_samples_per_prompt > 1:
            group_reward_stds = (
                rewards.std(-1, keepdim=True).repeat(1, args.n_samples_per_prompt).reshape(-1)[indices].split(exp_len)
            )
            for experience, group_reward_std in zip(experiences, group_reward_stds):
                experience.info["group_reward_std"] = group_reward_std

        # reward shaping
        if args.advantage_estimator == "rloo":
            baseline = (rewards.sum(-1, keepdim=True) - rewards) / (args.n_samples_per_prompt - 1)
            rewards = rewards - baseline
        elif args.advantage_estimator in ["reinforce_baseline", "dr_grpo"]:
            # REINFORCE++-baseline and Dr. GRPO removed the `/std` in GRPO as `/ std` is not needed in RL variance reduction theory.
            # And `k3 KL` has a larger variance than `k1 KL` under a categorical distribution.
            rewards = rewards - rewards.mean(-1, keepdim=True)
        elif args.advantage_estimator == "group_norm":
            rewards = (rewards - rewards.mean(-1, keepdim=True)) / (rewards.std(-1, keepdim=True) + 1e-9)

        rewards = rewards.reshape(-1)[indices].split(exp_len)

        # calculate return and advantages
        for experience, reward in zip(experiences, rewards):
            reward = compute_reward(
                reward,
                self.kl_ctl.value,
                experience.kl,
                action_mask=experience.action_mask,
                reward_clip_range=args.reward_clip_range,
            )

            if self.advantage_estimator == "gae":
                experience.advantages, experience.returns = self.get_advantages_and_returns(
                    experience.values,
                    reward,
                    experience.action_mask,
                    args.gamma,
                    args.lambd,
                )
            elif self.advantage_estimator in ["reinforce", "rloo", "reinforce_baseline", "group_norm", "dr_grpo"]:
                if args.gamma != 1.0 and self.advantage_estimator in [
                    "rloo",
                    "reinforce_baseline",
                    "group_norm",
                    "dr_grpo",
                ]:
                    logger.warning("gamma is set to 1.0 for rloo, reinforce_baseline, and group_norm")
                    args.gamma = 1.0

                experience.returns = self.get_cumulative_returns(
                    reward,
                    experience.action_mask,
                    args.gamma,
                )
                experience.advantages = deepcopy(experience.returns)
            else:
                raise Exception(f"Unkown advantage_estimator {self.advantage_estimator}")

            # calculate the return info.
            return_sums = reward.sum(dim=-1)
            experience.info["return"] = return_sums
            # remove unnecessary info
            experience.kl = None

        # Normalize advantages across all experiences for GAE, REINFORCE, and REINFORCE-baseline
        if self.args.advantage_estimator in ["gae", "reinforce", "reinforce_baseline"]:
            all_advantages = []
            all_action_masks = []
            for exp in experiences:
                all_advantages.append(exp.advantages.flatten())
                all_action_masks.append(exp.action_mask.flatten())

            advantages_vector = torch.cat(all_advantages, dim=0).float()
            action_masks_vector = torch.cat(all_action_masks, dim=0)
            num_actions = action_masks_vector.sum()

            # mean
            mean = (advantages_vector * action_masks_vector).sum() / num_actions
            # std
            if not self.args.no_advantage_std_norm:
                var = ((advantages_vector - mean).pow(2) * action_masks_vector).sum() / num_actions
                rstd = var.clamp(min=1e-8).rsqrt()
            else:
                rstd = 1

            # Apply normalization to each experience
            for exp in experiences:
                exp.advantages = (exp.advantages - mean) * rstd

        return experiences

    @torch.no_grad()
    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Input:
        - values: Tensor of shape (batch_size, response_size)
        - rewards: Tensor of shape (batch_size, response_size)

        Output:
        - advantages: Tensor of shape (batch_size, response_size)
        - returns: Tensor of shape (batch_size, response_size)
        """
        lastgaelam = 0
        advantages_reversed = []
        response_length = rewards.size(1)

        # Mask invalid responses
        if action_mask is not None:
            values = action_mask * values
            rewards = action_mask * rewards

        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages.detach(), returns

    @torch.no_grad()
    def get_cumulative_returns(
        self,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Function that computes advantages and returns from rewards using REINFORCE.
        REINFORCE uses cumulative returns without the GAE (Generalized Advantage Estimation).

        Input:
        - rewards: Tensor of shape (batch_size, response_size)
        - action_mask: Tensor of shape (batch_size, response_size), binary mask
        - gamma: discount factor

        Output:
        - returns: Tensor of shape (batch_size, response_size)
        """
        response_length = rewards.size(1)
        returns = torch.zeros_like(rewards)
        cumulative_return = torch.zeros(rewards.size(0), device=rewards.device)

        # Mask invalid responses if action_mask is provided
        if action_mask is not None:
            rewards = action_mask * rewards

        # Calculate returns by accumulating discounted rewards
        for t in reversed(range(response_length)):
            cumulative_return = rewards[:, t] + gamma * cumulative_return
            returns[:, t] = cumulative_return

        return returns
