import random
import time
from abc import ABC, abstractmethod
from typing import List, Optional

import ray
import torch

from openrlhf.trainer.ppo_utils.experience_maker import Experience, SamplesGenerator
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
        is_valid = self.min_r + 1e-6 < avg_reward < self.max_r - 1e-6
        if is_valid:
            self.valid_groups += 1
            return experiences
        return []

    def pass_rate(self) -> Optional[float]:
        if self.total_groups == 0:
            return None
        return self.valid_groups / self.total_groups * 100

    def reset(self):
        self.total_groups = 0
        self.valid_groups = 0


def _collect_prompts(target_prompts: int, dataloader_iter):
    prompts, labels = [], []
    exhausted = False

    while len(prompts) < target_prompts:
        try:
            _, rand_prompts, rand_labels = next(dataloader_iter)
            remaining = target_prompts - len(prompts)
            prompts.extend(rand_prompts[:remaining])
            labels.extend(rand_labels[:remaining])
        except StopIteration:
            exhausted = True
            break

    return prompts, labels, exhausted


def _filter_batches(samples: List[Experience], group: int, hook: FilterHookBase) -> List[Experience]:
    filtered: List[Experience] = []
    for i in range(0, len(samples), group):
        batch = samples[i : i + group]
        kept = hook.apply(batch)
        filtered.extend(kept)
    return filtered


class SamplesGeneratorAsync(SamplesGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_batch(self, target_prompts: int, dataloader_iter, **kwargs) -> tuple[List[Experience], bool, int]:
        filter_hook: FilterHookBase = kwargs.pop("filter_hook", NoOpFilterHook())

        prompts, labels, exhausted = _collect_prompts(target_prompts, dataloader_iter)
        if not prompts:
            return [], True, 0

        samples = super().generate_samples(prompts, labels, **kwargs)
        filtered = _filter_batches(samples, self.args.n_samples_per_prompt, filter_hook)
        return filtered, exhausted, len(prompts)

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
            logprobs=1 if args.enable_vllm_is_correction else None,
        )
        truncate_length = self.prompt_max_len + kwargs.get("max_new_tokens", 1024)

        # Expand prompt list based on the number of samples per prompt
        n_samples_per_prompt = kwargs.pop("n_samples_per_prompt", args.n_samples_per_prompt)
        all_prompts = sum([[prompt] * n_samples_per_prompt for prompt in all_prompts], [])
        all_labels = sum([[label] * n_samples_per_prompt for label in all_labels], [])

        # Distribute requests to engines and collect responses to outputs
        refs = []
        batch_size = (len(all_prompts) + len(llms) - 1) // len(llms)
        for i, llm in enumerate(llms):
            prompts = all_prompts[i * batch_size : (i + 1) * batch_size]
            labels = all_labels[i * batch_size : (i + 1) * batch_size]
            refs.append(
                llm.add_requests.remote(
                    sampling_params=sampling_params,
                    prompts=prompts,
                    labels=labels,
                    max_length=truncate_length,
                    hf_tokenizer=self.tokenizer,
                )
            )
        ray.get(refs)

        # Retrieve and combine results from all outputs
        all_output_refs = []
        for i, llm in enumerate(llms):
            all_output_refs.append(llm.get_responses.remote())
        all_outputs = sum(ray.get(all_output_refs), [])

        # Group outputs by prompt
        prompt_groups = {}
        for output in all_outputs:
            prompt = output["prompt"]
            prompt_groups.setdefault(prompt, []).append(output)

        # Reorder outputs to keep same prompts together
        # This is very important for REINFORCE++-baseline/GRPO/RLOO
        all_outputs = []
        for prompt in prompt_groups.keys():
            all_outputs.extend(prompt_groups[prompt])

        # Process outputs one by one
        experiences_list = []
        for output in all_outputs:
            # Get observation tokens directly (already tokenized)
            observation_tokens = output["observation_tokens"]
            tokenized_observation = observation_tokens.copy()

            # Action ranges are already in token space
            tokenized_ranges = output["action_ranges"]

            # Create tensors
            sequences = torch.tensor(tokenized_observation, dtype=torch.long)
            attention_mask = torch.tensor([1] * len(tokenized_observation))

            # Create action mask based on tokenized action_ranges
            action_mask = torch.zeros_like(attention_mask)
            # Mark action positions in the mask
            for start, end in tokenized_ranges:
                action_mask[start:end] = 1

            # Apply length limit
            sequences = sequences[:truncate_length].to("cpu")
            attention_mask = attention_mask[:truncate_length].to("cpu")
            action_mask = action_mask[1:truncate_length].to("cpu")
            if output["rollout_log_probs"] is not None:
                rollout_log_probs = torch.tensor(output["rollout_log_probs"][1:truncate_length]).to("cpu")
            else:
                rollout_log_probs = None

            # Calculate response length (distance between first and last 1)
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
            experiences_list.append(experience)

        return experiences_list


class SamplesGeneratorStreamingAsync(SamplesGeneratorAsync):
    """
    Streaming sampler based on the OpenRLHF Agent architecture.
    Uses LLMRayActorAsync and result_queue to implement prompt-level asynchronous gathering.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_batch(self, target_prompts: int, dataloader_iter, **kwargs) -> tuple[List[Experience], bool, int]:
        """
        Core method for streaming sampling.

        Args:
            target_prompts: The target number of prompts.
            dataloader_iter: Iterator over the prompt dataloader (stateful across calls).

        Returns:
            A tuple containing the list of samples, whether the dataset is exhausted, and the filter pass rate.
        """
        if dataloader_iter is None:
            raise ValueError("dataloader_iter must be provided for streaming sampling")

        filter_hook: FilterHookBase = kwargs.pop("filter_hook", NoOpFilterHook())

        valid_experiences = []
        active_requests = {}  # {request_id: request_info}
        dataloader_exhausted = False  # Track dataloader status

        # Calculate the target number of samples for internal logic
        target_samples = target_prompts * self.args.n_samples_per_prompt

        # Track dataset status
        self._dataloader_consumed_count = 0
        self._total_prompt_processed = 0
        self._valid_prompt_collected = 0

        logger.info(f"Starting streaming sampling, target: {target_prompts} prompts")

        # 1. Initialize: Assign initial prompts to each engine
        try:
            initial_requests = self._get_initial_request_pool(dataloader_iter)
        except StopIteration:
            logger.warning("Dataloader exhausted during initialization")
            dataloader_exhausted = True
            initial_requests = []

        # 2. Submit initial requests to LLMRayActorAsync
        for engine_idx, prompt, label in initial_requests:
            request_id = f"prompt_{time.time()}_{random.randint(1000, 9999)}"
            engine = self.vllm_engines[engine_idx]

            # Use the existing LLMRayActorAsync.add_requests
            ref = engine.add_requests.remote(
                sampling_params=self._build_sampling_params(**kwargs),
                prompts=[prompt] * self.args.n_samples_per_prompt,  # Replicate prompt
                labels=[label] * self.args.n_samples_per_prompt,  # Replicate label
                max_length=self.prompt_max_len + kwargs.get("max_new_tokens", 1024),
                hf_tokenizer=self.tokenizer,
                request_group_id=request_id,
            )

            active_requests[request_id] = {
                "engine_ref": ref,
                "engine_idx": engine_idx,
                "engine": engine,
                "prompt": prompt,
                "label": label,
            }

        logger.info(f"Submitted {len(active_requests)} initial requests to {len(self.vllm_engines)} engines")

        # 3. Stream-collect completed prompt groups using the standard ray.wait pattern
        # Build a map from refs to request_id
        refs_to_request_id = {
            request_info["engine_ref"]: request_id for request_id, request_info in active_requests.items()
        }
        remaining_refs = list(refs_to_request_id.keys())

        while len(valid_experiences) < target_samples and remaining_refs:
            # If the dataloader is exhausted and there are no remaining requests, exit early
            if dataloader_exhausted and not remaining_refs:
                logger.warning(
                    f"Dataloader exhausted and no remaining requests. Collected {len(valid_experiences)}/{target_samples} samples."
                )
                break

            try:
                # Standard ray.wait pattern: wait for any request to complete
                ready_refs, remaining_refs = ray.wait(remaining_refs, num_returns=1, timeout=10.0)

                # Process all completed requests
                for completed_ref in ready_refs:
                    request_id = refs_to_request_id[completed_ref]
                    request_info = active_requests.pop(request_id)

                    try:
                        # Wait for add_requests to complete
                        ray.get(completed_ref)

                        # Get the Agent execution results for the specific request_group
                        agent_outputs = ray.get(request_info["engine"].get_responses.remote(request_id))

                        # Check if the number of results is correct
                        if len(agent_outputs) != self.args.n_samples_per_prompt:
                            logger.warning(
                                f"Invalid group size: got {len(agent_outputs)}, expected {self.args.n_samples_per_prompt}"
                            )
                            # Skip invalid result groups
                            continue

                        # Convert to Experience objects
                        prompt_experiences = self._convert_agent_outputs_to_experiences(agent_outputs, request_info)

                        kept = filter_hook.apply(prompt_experiences)

                        if kept:
                            valid_experiences.extend(kept)
                            self._valid_prompt_collected += 1  # Increment valid prompt count

                            # Precise stop check
                            if len(valid_experiences) >= target_samples:
                                # Cancel remaining requests
                                self._cancel_remaining_requests(remaining_refs, refs_to_request_id, active_requests)
                                logger.info(
                                    f"Target reached! Collected {self._valid_prompt_collected} prompts ({len(valid_experiences)} samples), cancelling {len(remaining_refs)} remaining requests"
                                )
                                return (
                                    valid_experiences[:target_samples],
                                    False,
                                    self._total_prompt_processed,
                                )  # Target reached, but dataset is not exhausted
                        else:
                            # Invalid group: dynamically supplement with a new request
                            new_ref, new_request_id, new_request_info = self._supplement_new_request_optimized(
                                dataloader_iter, dataloader_exhausted, **kwargs
                            )
                            if new_ref is not None:
                                # Add the new request for tracking
                                remaining_refs.append(new_ref)
                                refs_to_request_id[new_ref] = new_request_id
                                active_requests[new_request_id] = new_request_info
                            else:
                                dataloader_exhausted = True

                    except Exception as e:
                        logger.error(f"Error processing request {request_id}: {e}")
                        # Also supplement for failed requests
                        if not dataloader_exhausted:
                            new_ref, new_request_id, new_request_info = self._supplement_new_request_optimized(
                                dataloader_iter, dataloader_exhausted, **kwargs
                            )
                            if new_ref is not None:
                                remaining_refs.append(new_ref)
                                refs_to_request_id[new_ref] = new_request_id
                                active_requests[new_request_id] = new_request_info
                            else:
                                dataloader_exhausted = True

            except Exception as e:
                logger.error(f"Error in ray.wait: {e}")
                # If ray.wait fails, try to process the first request
                if remaining_refs:
                    remaining_refs = remaining_refs[1:]  # Remove the potentially problematic ref

        # Process final results
        final_count = len(valid_experiences)
        final_prompts = self._valid_prompt_collected
        logger.info("Final statistics:")
        logger.info(
            f"   - Target prompts: {target_prompts} ({target_prompts}x{self.args.n_samples_per_prompt}={target_samples} samples)"
        )
        logger.info(f"   - Collected prompts: {final_prompts} ({final_count} samples)")
        logger.info(f"   - Dataloader batches consumed: {self._dataloader_consumed_count}")
        logger.info(f"   - Total prompts processed: {self._total_prompt_processed}")
        logger.info(
            f"   - Prompt success rate: {final_prompts / target_prompts * 100:.1f}% ({final_count / target_samples * 100:.1f}% samples)"
        )

        filter_pass_rate = filter_hook.pass_rate()
        if filter_pass_rate is not None:
            logger.info(f"   - Filter pass rate: {filter_pass_rate:.1f}%")

        if final_count < target_samples:
            logger.warning(
                f"Could not collect target. Collected {final_prompts}/{target_prompts} prompts ({final_count}/{target_samples} samples)"
            )
            if dataloader_exhausted:
                logger.info("Hint: consider adjusting dynamic_filtering_reward_range or increasing dataset size")
                logger.info(f"Hint: dataloader was exhausted after {self._dataloader_consumed_count} batches")
        else:
            logger.info(f"Streaming sampling completed: {final_prompts} prompts collected ({final_count} samples)")

        # Calculate the final filter pass rate
        final_filter_pass_rate = filter_hook.pass_rate()

        return (
            (valid_experiences[:target_samples] if final_count >= target_samples else valid_experiences),
            dataloader_exhausted,
            self._total_prompt_processed,
        )

    def generate_samples(
        self, prompts: List[str], labels: List[str], filter_hook: Optional[FilterHookBase] = None, **kwargs
    ) -> List[Experience]:
        """Compatibility wrapper; streams over a one-off iterator."""
        dataloader_iter = iter([(None, prompts, labels)])
        target_prompts = len(prompts)
        samples, _, _ = self.generate_batch(
            target_prompts=target_prompts,
            dataloader_iter=dataloader_iter,
            filter_hook=filter_hook or NoOpFilterHook(),
            **kwargs,
        )
        return samples

    def _get_initial_request_pool(self, dataloader_iter):
        """Get the initial request pool, distributed by the number of engines."""
        initial_requests = []

        # Calculate initial load for each engine
        batch_size = self.args.rollout_batch_size
        engine_count = len(self.vllm_engines)

        # Get enough prompts from the dataloader
        collected_prompts = []
        collected_labels = []

        logger.info(f"Initializing request pool, need {batch_size} prompts for {engine_count} engines")

        while len(collected_prompts) < batch_size:
            try:
                _, rand_prompts, labels = next(dataloader_iter)
                self._dataloader_consumed_count += 1
                collected_prompts.extend(rand_prompts)
                collected_labels.extend(labels)
            except StopIteration:
                logger.warning(
                    f"Dataloader exhausted during initialization after {self._dataloader_consumed_count} fetches"
                )
                break

        # Assign prompts to engines
        for i in range(min(batch_size, len(collected_prompts))):
            engine_idx = i % engine_count
            initial_requests.append((engine_idx, collected_prompts[i], collected_labels[i]))
            self._total_prompt_processed += 1

        logger.info(
            f"Initial request pool created: {len(initial_requests)} requests, consumed {self._dataloader_consumed_count} dataloader batches"
        )
        return initial_requests

    def _convert_agent_outputs_to_experiences(self, agent_outputs, request_info):
        """
        Convert Agent execution results to Experience objects.
        Reuses existing conversion logic.
        """
        experiences = []

        for output in agent_outputs:
            try:
                # Agent output format is compatible with the current output format
                # Reuse existing Experience creation logic
                experience = self._create_experience_from_output(output, request_info["prompt"], request_info["label"])
                experiences.append(experience)
            except Exception as e:
                logger.error(f"Error converting agent output to experience: {e}")

        return experiences

    def _create_experience_from_output(self, output, prompt, label):
        """
        Create an Experience object from a single output.
        Reuses existing logic to maintain consistency with current experience creation.
        """
        # Get observation tokens
        observation_tokens = output["observation_tokens"]
        tokenized_observation = observation_tokens.copy()

        # Action ranges are already in token space
        tokenized_ranges = output["action_ranges"]

        # Create tensors
        sequences = torch.tensor(tokenized_observation)
        attention_mask = torch.tensor([1] * len(tokenized_observation))

        # Create action mask based on tokenized action_ranges
        action_mask = torch.zeros_like(attention_mask)
        for start, end in tokenized_ranges:
            action_mask[start:end] = 1

        # Apply length limit
        truncate_length = self.prompt_max_len + 1024  # Default truncation length
        sequences = sequences[:truncate_length].to("cpu")
        attention_mask = attention_mask[:truncate_length].to("cpu")
        action_mask = action_mask[1:truncate_length].to("cpu")

        if output["rollout_log_probs"] is not None:
            rollout_log_probs = torch.tensor(output["rollout_log_probs"][1:truncate_length]).to("cpu")
        else:
            rollout_log_probs = None

        # Calculate response length
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
            prompts=[prompt],
            labels=[label],
            rewards=torch.tensor([output["reward"]]),
            scores=torch.tensor([output["scores"]]),
            info=info,
        )

        return experience

    def _supplement_new_request_optimized(self, dataloader_iter, dataloader_exhausted, **kwargs):
        """
        Optimized method to supplement requests, directly returning the new ref, request_id, and request_info.

        Returns:
            A tuple (new_ref, new_request_id, new_request_info) on success, or (None, None, None) if the dataloader is exhausted.
        """
        if dataloader_exhausted:
            logger.debug(
                f"Skip supplement: dataloader already exhausted (consumed {self._dataloader_consumed_count} batches)"
            )
            return None, None, None

        try:
            _, rand_prompts, labels = next(dataloader_iter)
            self._dataloader_consumed_count += 1
            self._total_prompt_processed += 1

            # Use the first prompt and label for supplementation
            new_prompt, new_label = rand_prompts[0], labels[0]
            best_engine_idx = self._select_best_engine()
            engine = self.vllm_engines[best_engine_idx]

            logger.info(
                f"Supplement #{self._dataloader_consumed_count}: got {len(rand_prompts)} prompts, total processed: {self._total_prompt_processed}"
            )

            new_request_id = f"prompt_{time.time()}_{random.randint(1000, 9999)}"
            prompts_to_submit = [new_prompt] * self.args.n_samples_per_prompt
            labels_to_submit = [new_label] * self.args.n_samples_per_prompt
            logger.debug(f"Submitting {len(prompts_to_submit)} prompts to engine {best_engine_idx}")

            new_ref = engine.add_requests.remote(
                sampling_params=self._build_sampling_params(**kwargs),
                prompts=prompts_to_submit,
                labels=labels_to_submit,
                max_length=self.prompt_max_len + kwargs.get("max_new_tokens", 1024),
                hf_tokenizer=self.tokenizer,
                request_group_id=new_request_id,  # Pass request_group_id for result isolation
            )

            # Build the request info object
            new_request_info = {
                "engine_ref": new_ref,
                "engine_idx": best_engine_idx,
                "engine": engine,
                "prompt": new_prompt,
                "label": new_label,
            }

            return new_ref, new_request_id, new_request_info

        except StopIteration:
            logger.warning(
                f"Dataloader exhausted, cannot supplement more prompts (total consumed: {self._dataloader_consumed_count} batches, processed: {self._total_prompt_processed} prompts)"
            )
            return None, None, None

    def _select_best_engine(self):
        """
        Selects the engine with the lightest load.
        A round-robin strategy ensures requests are distributed across different engines, avoiding queue contention.
        """
        if not hasattr(self, "_engine_counter"):
            self._engine_counter = 0
        engine_idx = self._engine_counter % len(self.vllm_engines)
        self._engine_counter += 1
        return engine_idx

    def _cancel_remaining_requests(self, remaining_refs, refs_to_request_id, active_requests):
        """
        Cancel remaining requests to achieve precise stopping.
        """
        cancelled_count = 0
        for ref in remaining_refs:
            try:
                # Try to cancel the Ray task
                ray.cancel(ref)
                request_id = refs_to_request_id.get(ref, "unknown")
                logger.debug(f"Cancelled request {request_id}")
                cancelled_count += 1

                # Remove from active_requests
                if request_id in active_requests:
                    active_requests.pop(request_id)

            except Exception as e:
                request_id = refs_to_request_id.get(ref, "unknown")
                logger.warning(f"Failed to cancel request {request_id}: {e}")

        logger.info(f"Cancelled {cancelled_count} remaining requests")

    def _build_sampling_params(self, **kwargs):
        """
        Build sampling parameters.
        """
        from vllm import SamplingParams

        return SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", 1024),
            min_tokens=kwargs.get("min_new_tokens", 1),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
            logprobs=1 if self.strategy.args.enable_vllm_is_correction else None,
        )
