import time
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from datetime import timedelta
from typing import List, Tuple, Union

import ray
import torch
import torch.distributed as dist

from openrlhf.models.utils import compute_approx_kl, compute_reward, masked_mean
from openrlhf.trainer.ray.launcher import PPORayActorGroup
from openrlhf.utils.logging_utils import init_logger

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
    All fields are stored as lists, where tensor fields are stored as lists of tensors.
    Left padding for sequences is applied.

    Shapes of each tensor in the lists:
    sequences: (B, S)
    action_log_probs: (B, A)
    base_action_log_probs: (B, A)
    values: (B, A)
    returns: (B, A)
    advantages: (B, A)
    attention_mask: (B, S)
    action_mask: (B, A)
    response_length: (B,)
    total_length: (B,)
    kl: (B, A)

    "A" is the number of actions.
    """

    # All fields are lists
    prompts: list[str] = None
    labels: list[str] = None
    rewards: list[float] = None
    scores: list[float] = None

    sequences: list[torch.Tensor] = None
    attention_mask: list[torch.LongTensor] = None
    action_mask: list[torch.BoolTensor] = None
    response_length: list[torch.Tensor] = None
    total_length: list[torch.Tensor] = None

    action_log_probs: list[torch.Tensor] = None
    base_action_log_probs: list[torch.Tensor] = None
    values: list[torch.Tensor] = None
    returns: list[torch.Tensor] = None
    advantages: list[torch.Tensor] = None
    kl: list[torch.Tensor] = None
    info: list[dict] = None

    def __init__(
        self,
        sequences=None,
        action_log_probs=None,
        base_action_log_probs=None,
        values=None,
        returns=None,
        advantages=None,
        attention_mask=None,
        action_mask=None,
        response_length=None,
        total_length=None,
        kl=None,
        prompts=None,
        labels=None,
        rewards=None,
        info=None,
    ):
        self.sequences = sequences or []
        self.action_log_probs = action_log_probs or []
        self.base_action_log_probs = base_action_log_probs or []
        self.values = values or []
        self.returns = returns or []
        self.advantages = advantages or []
        self.attention_mask = attention_mask or []
        self.action_mask = action_mask or []
        self.response_length = response_length or []
        self.total_length = total_length or []
        self.kl = kl or []
        self.prompts = prompts or []
        self.labels = labels or []
        self.rewards = rewards or []
        self.info = info or []

    @torch.no_grad()
    def to_device(self, device: torch.device):
        """Move all tensor fields to the specified device."""
        tensor_fields = [
            "sequences",
            "action_log_probs",
            "base_action_log_probs",
            "values",
            "returns",
            "advantages",
            "attention_mask",
            "action_mask",
            "response_length",
            "total_length",
            "kl",
        ]

        for field in tensor_fields:
            value = getattr(self, field)
            if value:
                setattr(self, field, [to(t, device) for t in value])

        if self.info:
            self.info = [{key: to(value, device) for key, value in info_dict.items()} for info_dict in self.info]

        return self

    def pin_memory(self):
        """Pin memory for all tensor fields."""
        tensor_fields = [
            "sequences",
            "action_log_probs",
            "base_action_log_probs",
            "values",
            "returns",
            "advantages",
            "attention_mask",
            "action_mask",
            "response_length",
            "total_length",
            "kl",
        ]

        for field in tensor_fields:
            value = getattr(self, field)
            if value:
                setattr(self, field, [pin_memory(t) for t in value])

        if self.info:
            self.info = [{key: pin_memory(value) for key, value in info_dict.items()} for info_dict in self.info]

        return self

    @staticmethod
    def _merge_dicts(dicts: List[dict]) -> dict:
        """Merge a list of dictionaries into a single dictionary.
        If there are duplicate keys, the last value will be used.
        """
        if not dicts:
            return {}

        result = {}
        for d in dicts:
            result.update(d)
        return result

    @staticmethod
    def concat_experiences(experiences_list: List["Experience"], pad_token_id) -> "Experience":
        """Concatenate multiple experiences into one large experience.

        Args:
            experiences_list: List of Experience to concatenate

        Returns:
            A new Experience instance containing all the concatenated data
        """
        if not experiences_list:
            return Experience()

        # Get all field names from the dataclass
        fields = [f for f in Experience.__dataclass_fields__]

        # Create result dictionary
        result = {}

        # Merge all fields
        for field in fields:
            if field == "info":
                # Special handling for info field: merge dictionaries
                result[field] = [Experience._merge_dicts([d for e in experiences_list for d in e.info])]
            else:
                # For other fields: concatenate lists
                result[field] = sum([getattr(e, field) for e in experiences_list], [])

        return Experience(**result)


class SamplesGenerator:
    def __init__(self, vllm_engines, strategy, tokenizer, prompt_max_len):
        self.strategy = strategy
        self.args = strategy.args
        self.vllm_engines = vllm_engines
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len

    @torch.no_grad()
    def generate_samples(self, all_prompts: List[str], all_labels, **generate_kwargs) -> List[Experience]:
        """
        Generate samples and return in batches.

        When not using vllm, we will fallback to the default implementation,
        in which actor will be used to generate samples.
        """
        # vLLM wakeup when vllm_enable_sleep
        if self.strategy.args.vllm_enable_sleep:
            from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call

            batch_vllm_engine_call(self.vllm_engines, "wake_up")

        rollout_samples = self._generate_vllm(all_prompts, all_labels, **generate_kwargs)

        # vLLM offload when vllm_enable_sleep
        if self.strategy.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "sleep")

        return rollout_samples

        # tokenizer

    def tokenize_fn(self, texts, max_length, padding=True, device=None):
        if not padding:
            # when padding is False, return tokenized texts as list
            return self.tokenizer(
                texts,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
            )
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

    def _generate_vllm(self, all_prompts: List[str], all_labels, **kwargs) -> List[Experience]:
        from vllm import SamplingParams

        llms = self.vllm_engines
        args = self.strategy.args

        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", 1024),
            min_tokens=kwargs.get("min_new_tokens", 1),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
            include_stop_str_in_output=True,
        )
        truncate_length = self.prompt_max_len + kwargs.get("max_new_tokens", 1024)

        # Expand prompt list based on the number of samples per prompt
        n_samples_per_prompt = kwargs.pop("n_samples_per_prompt", args.n_samples_per_prompt)
        all_prompts = sum([[prompt] * n_samples_per_prompt for prompt in all_prompts], [])
        all_labels = sum([[label] * n_samples_per_prompt for label in all_labels], [])
        all_prompt_token_ids = self.tokenize_fn(all_prompts, self.prompt_max_len, padding=False)["input_ids"]

        # Distribute requests to engines and collect responses to outputs
        refs = []
        batch_size = (len(all_prompt_token_ids) + len(llms) - 1) // len(llms)
        for i, llm in enumerate(llms):
            prompt_token_ids = all_prompt_token_ids[i * batch_size : (i + 1) * batch_size]
            refs.append(llm.add_requests.remote(sampling_params=sampling_params, prompt_token_ids=prompt_token_ids))
        ray.get(refs)

        # Retrieve and combine results from all outputs
        all_output_refs = []
        for i, llm in enumerate(llms):
            all_output_refs.append(llm.get_responses.remote())
        all_outputs = sum(ray.get(all_output_refs), [])

        pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id

        # Group outputs by n_samples_per_prompt
        samples_list = []
        for i in range(len(all_outputs)):
            output = all_outputs[i]
            prompt = all_prompts[i]
            label = all_labels[i]

            # Concatenate prompt and output tokens
            input_ids = list(output.prompt_token_ids) + list(output.outputs[0].token_ids)
            if output.outputs[0].token_ids[-1] != eos_token_id:
                input_ids.append(eos_token_id)
            # Create attention mask
            attention_mask = [1] * len(input_ids)

            sequences = torch.tensor(input_ids)
            attention_mask = torch.tensor(attention_mask)

            # Create action mask based on output token positions
            action_mask = torch.zeros_like(attention_mask)
            # Mark positions after prompt as actions
            action_length = len(output.outputs[0].token_ids) + int(output.outputs[0].token_ids[-1] != eos_token_id)
            action_mask[len(output.prompt_token_ids) : len(output.prompt_token_ids) + action_length] = 1

            sequences = sequences[:truncate_length].to("cpu")
            attention_mask = attention_mask[:truncate_length].to("cpu")
            action_mask = action_mask[1:truncate_length].to("cpu")
            response_length = action_mask.float().sum().item()
            total_length = attention_mask.float().sum().item()

            rollout_samples = Experience(
                sequences=[sequences],
                attention_mask=[attention_mask],
                action_mask=[action_mask],
                response_length=[response_length],
                total_length=[total_length],
                prompts=[prompt],
                labels=[label],
            )
            samples_list.append(rollout_samples)

        # Get rewards from remote reward models
        # This is required by dynamic sampling
        remote_reward_model = kwargs.get("remote_reward_model", None)
        if remote_reward_model:
            all_queries = sum(
                [self.tokenizer.batch_decode(s.sequences[0], skip_special_tokens=False) for s in samples_list], []
            )
            all_prompts = sum([s.prompts for s in samples_list], [])
            all_labels = sum([s.labels for s in samples_list], [])

            rewards_list = ray.get(remote_reward_model.get_rewards.remote(all_queries, all_prompts, all_labels))
            rewards_list = torch.cat(rewards_list, dim=0).chunk(len(samples_list))

            for i, samples in enumerate(samples_list):
                samples.rewards = rewards_list[i].tolist()

        return samples_list


class RemoteExperienceMaker(ABC):
    def __init__(
        self,
        actor_model_group: PPORayActorGroup,
        critic_model_group: PPORayActorGroup,
        reward_model_group: PPORayActorGroup,
        initial_model_group: PPORayActorGroup,
        kl_controller,
        strategy=None,
        tokenizer=None,
        remote_reward_model=None,
        **kwargs,
    ):
        super().__init__()

        self.actor_model_group = actor_model_group
        self.critic_model_group = critic_model_group
        self.reward_model_group = reward_model_group
        self.initial_model_group = initial_model_group
        self.kl_ctl = kl_controller
        self.strategy = strategy
        self.advantage_estimator = strategy.args.advantage_estimator
        self.args = strategy.args

        # remote_rm_url indicates that the remote reward model is agent enviroment, remote http server or custom reward func
        self.remote_rm_url = self.args.remote_rm_url
        self.remote_reward_model = remote_reward_model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def make_experience_batch(self, rollout_samples) -> List[Experience]:
        """
        Make a list of experience with the micro_rollout_batch_size.

        This method will first calculate the response sequences and rewards for the given prompts.
        Then, if we need certain processing for the rewards or do certain filtering, we can process the rollout as a whole.
        After that, we will calculate the advantages and returns for each experience.
        """
        # Concat the samples into micro_rollout_batch_size
        # Each batch of samples will be scheduled to a effective Ray Actor (i.e, a DP rank)
        # TODO: balance the number of tokens of each batch for better performance
        samples_list = []
        # balance the number of tokens of each batch for better performance
        if self.args.use_dynamic_batch_size:
            dp_size = dist.get_world_size(group=self.strategy.ds_device_mesh["dp"].get_group())

            # calculate token count for each sample and sort by token count
            sample_token_counts = []
            for sample in rollout_samples:
                token_count = sample.total_length[0].item()
                sample_token_counts.append((token_count, sample))

            # Sort samples by token count in descending order
            sample_token_counts.sort(key=lambda x: x[0], reverse=True)

            # Initialize batches
            batches = [[] for _ in range(dp_size)]
            batch_token_counts = [0] * dp_size

            # Assign samples to batches
            for token_count, sample in sample_token_counts:
                # Find the batch with minimum tokens
                min_batch_idx = batch_token_counts.index(min(batch_token_counts))
                batches[min_batch_idx].append(sample)
                batch_token_counts[min_batch_idx] += token_count

            # Concatenate samples in each batch
            for batch in batches:
                if batch:
                    concat_samples = Experience.concat_experiences(batch, self.tokenizer.pad_token_id)
                    samples_list.append(concat_samples)
        else:
            batch_size = self.args.micro_rollout_batch_size
            for i in range(0, len(rollout_samples), batch_size):
                concat_samples = Experience.concat_experiences(
                    rollout_samples[i : i + batch_size], self.tokenizer.pad_token_id
                )
                samples_list.append(concat_samples)

        # Make experiences (models forward: logprobs, values, rewards, and kl divergence)
        experiences = self.make_experience(samples_list)

        # Process experiences (reward shaping, etc.)
        experiences = self.compute_advantages_and_returns(experiences)
        return experiences

    @torch.no_grad()
    def make_experience(self, experiences_list: List[Experience]) -> List[Experience]:
        """
        Turn experiences into complete experiences by calculating logprobs, values, rewards, and kl divergence.
        The input experiences already contain basic information like sequences, attention_mask, etc.
        This method will add or update the following fields:
        - action_log_probs
        - base_action_log_probs
        - values
        - returns
        - advantages
        - kl
        - info
        """
        start_time = time.time()
        logger.info(f"🚀 Starting experience making with {len(experiences_list)} experiences")

        args = self.strategy.args
        device = "cpu"

        # Extract all information from experiences in one pass
        sequences_list = [e.sequences for e in experiences_list]
        attention_mask_list = [e.attention_mask for e in experiences_list]
        action_mask_list = [e.action_mask for e in experiences_list]

        # Get rewards from experiences, such as rewards from agent's environment
        if experiences_list[0].rewards:
            rewards_list = [e.rewards for e in experiences_list]
            rewards_list = [torch.tensor(s.rewards) for s in experiences_list]
            r_refs = ray.put(rewards_list)
        elif self.remote_rm_url:
            queries_list = sum(
                [self.tokenizer.batch_decode(seq, skip_special_tokens=False) for seq in sequences_list], []
            )
            prompts_list = sum([e.prompts for e in experiences_list], [])
            labels_list = sum([e.labels for e in experiences_list], [])
            r_refs = self.remote_reward_model.get_rewards.remote(queries_list, prompts_list, labels_list)
        else:
            # Batch call reward model
            r_refs = self.reward_model_group.async_run_method_batch(
                method_name="forward",
                sequences=sequences_list,
                attention_mask=attention_mask_list,
                pad_sequence=[True] * len(experiences_list),
            )

        # Sync to avoid GPU OOM when colocate models
        if args.colocate_all_models and not self.remote_rm_url:
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
            if args.colocate_critic_reward and not self.remote_rm_url:
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
            value_ref = ray.put(
                [[None]] * (len(experiences_list) * args.ring_attn_size * args.ds_tensor_parallel_size)
            )

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
                [[None]] * (len(experiences_list) * args.ring_attn_size * args.ds_tensor_parallel_size)
            )

        # Wait for all remote calls to complete and flatten the results
        # Note: the results duplicated ring_attn_size * ds_tensor_parallel_size times
        duplicate_factor = args.ring_attn_size * args.ds_tensor_parallel_size
        action_log_probs_list = sum(ray.get(action_log_probs_ref)[::duplicate_factor], [])
        base_action_log_probs_list = sum(ray.get(base_action_log_probs_ref)[::duplicate_factor], [])
        value_list = sum(ray.get(value_ref)[::duplicate_factor], [])
        rewards_list = ray.get(r_refs)
        if self.remote_rm_url is None:
            rewards_list = sum(rewards_list[::duplicate_factor], [])
        else:
            rewards_list = torch.cat(rewards_list, dim=0).chunk(len(experiences_list))

        assert (
            len(experiences_list)
            == len(action_log_probs_list)
            == len(base_action_log_probs_list)
            == len(value_list)
            == len(rewards_list)
        ), f"len(experiences_list): {len(experiences_list)}, len(action_log_probs_list): {len(action_log_probs_list)}, len(base_action_log_probs_list): {len(base_action_log_probs_list)}, len(value_list): {len(value_list)}, len(rewards_list): {len(rewards_list)}"

        # Process results for each experience
        for i, (experience, action_log_probs, base_action_log_probs, value, rewards) in enumerate(
            zip(experiences_list, action_log_probs_list, base_action_log_probs_list, value_list, rewards_list)
        ):
            # Calculate KL divergence for each item
            experience.action_log_probs = action_log_probs
            experience.base_action_log_probs = base_action_log_probs
            experience.values = value
            experience.rewards = rewards

            kl_list = []
            info = {"kl": [], "reward": [], "response_length": [], "total_length": []}

            for j in range(len(action_log_probs)):
                # Calculate KL
                if (self.initial_model_group is not None) and (not args.use_kl_loss):
                    kl = compute_approx_kl(
                        action_log_probs[j],
                        base_action_log_probs[j],
                        kl_estimator=self.strategy.args.kl_estimator,
                    )
                else:
                    kl = torch.zeros_like(action_log_probs[j], dtype=action_log_probs[j].dtype, device=device)

                kl_list.append(kl)

                # Calculate KL mean for info
                kl_mean = masked_mean(kl, experience.action_mask[j], dim=-1)

                # Append values to info lists
                info["kl"].append(kl_mean)
                info["reward"].append(rewards[j])
                info["response_length"].append(experience.response_length[j])
                info["total_length"].append(experience.total_length[j])

            experience.kl = kl_list
            experience.info = info

        end_time = time.time()
        duration = end_time - start_time
        time_str = str(timedelta(seconds=duration)).split(".")[0]
        logger.info(f"✨ Experience making completed in {time_str}")
        return experiences_list

    @torch.no_grad()
    def compute_advantages_and_returns(
        self, experiences: List[Experience], **kwargs
    ) -> Tuple[List[Experience], List[torch.Tensor]]:
        """
        Process experiences, this can be used to filter out some experiences or do some processing on the rewards.

        Output:
        - experiences: List of Experience
        - rewards: List of rewards
        """
        args = self.strategy.args

        # get rewards from experiences
        rewards = [experience.rewards for experience in experiences]

        # reward shaping
        if args.advantage_estimator == "rloo":
            rewards = torch.cat(rewards).reshape(-1, args.n_samples_per_prompt)
            baseline = (rewards.sum(-1, keepdim=True) - rewards) / (args.n_samples_per_prompt - 1)
            rewards = rewards - baseline
            rewards = rewards.reshape(-1).chunk(len(experiences))
        elif args.advantage_estimator in ["reinforce_baseline", "dr_grpo"]:
            # REINFORCE++-baseline and Dr. GRPO removed the `/std` in GRPO as `/ std` is not needed in RL variance reduction theory.
            # And `k3 KL` has a larger variance than `k1 KL` under a categorical distribution.
            rewards = torch.cat(rewards).reshape(-1, args.n_samples_per_prompt)
            rewards = rewards - rewards.mean(-1, keepdim=True)
            rewards = rewards.reshape(-1).chunk(len(experiences))
        elif args.advantage_estimator == "group_norm":
            rewards = torch.cat(rewards).reshape(-1, args.n_samples_per_prompt)
            rewards = (rewards - rewards.mean(-1, keepdim=True)) / (rewards.std(-1, keepdim=True) + 1e-9)
            rewards = rewards.reshape(-1).chunk(len(experiences))

        # calculate return and advantages
        for experience, reward in zip(experiences, rewards):
            reward = compute_reward(
                reward,
                self.kl_ctl.value,
                experience.kl[0],
                action_mask=experience.action_mask[0],
                reward_clip_range=args.reward_clip_range,
            )

            if self.advantage_estimator == "gae":
                experience.advantages[0], experience.returns[0] = self.get_advantages_and_returns(
                    experience.values[0],
                    reward,
                    experience.action_mask[0],
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

                experience.returns[0] = self.get_cumulative_returns(
                    reward,
                    experience.action_mask[0],
                    args.gamma,
                )
                experience.advantages[0] = deepcopy(experience.returns[0])
            else:
                raise Exception(f"Unkown advantage_estimator {self.advantage_estimator}")

            # calculate the return info.
            return_sums = reward.sum(dim=-1)
            experience.info[0]["return"] = return_sums
            # remove unnecessary info
            experience.kl[0] = None

        # Normalize advantages across all experiences
        if self.args.advantage_estimator not in ["group_norm", "dr_grpo"]:
            all_advantages = []
            all_action_masks = []
            for exp in experiences:
                all_advantages.append(exp.advantages[0].flatten())
                all_action_masks.append(exp.action_mask[0].flatten())

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
                exp.advantages[0] = (exp.advantages[0] - mean) * rstd

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
