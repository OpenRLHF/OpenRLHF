import asyncio
from abc import ABC, abstractmethod
from copy import deepcopy

import aiohttp

from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)
ABORT_FINISH_REASONS = {"abort", "aborted"}


def _extract_action_log_probs(generation_output):
    if generation_output.logprobs is None:
        return None

    action_log_probs = []
    for token_id, logprob_dict in zip(generation_output.token_ids, generation_output.logprobs, strict=False):
        token_logprob = logprob_dict.get(token_id)
        action_log_probs.append(token_logprob.logprob if token_logprob is not None else 0.0)
    return action_log_probs


async def _generate_action_with_partial_rollout(
    *,
    prompt_token_ids,
    sampling_params,
    max_length: int,
    hf_tokenizer,
    llm_engine,
    multi_modal_data=None,
):
    """Generate one model action, optionally resuming after aborts."""
    partial_rollout = getattr(llm_engine, "partial_rollout", False)
    original_max_tokens = getattr(sampling_params, "max_tokens", None)
    if original_max_tokens is None:
        original_max_tokens = max(1, max_length - len(prompt_token_ids))

    action_token_ids = []
    action_token_weight_versions = []
    action_log_probs = [] if getattr(sampling_params, "logprobs", None) is not None else None
    finish_reason = None

    while True:
        remaining_tokens = original_max_tokens - len(action_token_ids)
        if remaining_tokens <= 0:
            finish_reason = "length"
            break

        next_sampling_params = deepcopy(sampling_params)
        next_sampling_params.max_tokens = remaining_tokens
        chunk_weight_version = llm_engine.get_weight_version()
        request_output = await llm_engine.generate(
            prompt_token_ids + action_token_ids,
            next_sampling_params,
            multi_modal_data=multi_modal_data,
        )

        generation_output = request_output.outputs[0] if request_output.outputs else None
        if generation_output is None:
            finish_reason = None
            if partial_rollout:
                continue
            break

        chunk_token_ids = generation_output.token_ids
        finish_reason = generation_output.finish_reason

        if chunk_token_ids:
            action_token_ids.extend(chunk_token_ids)
            action_token_weight_versions.extend([chunk_weight_version] * len(chunk_token_ids))
            if action_log_probs is not None:
                chunk_log_probs = _extract_action_log_probs(generation_output)
                if chunk_log_probs is None:
                    chunk_log_probs = [0.0] * len(chunk_token_ids)
                action_log_probs.extend(chunk_log_probs)

        if finish_reason not in ABORT_FINISH_REASONS or not partial_rollout:
            break

    return {
        "token_ids": action_token_ids,
        "text": hf_tokenizer.decode(action_token_ids, skip_special_tokens=False),
        "log_probs": action_log_probs,
        "finish_reason": finish_reason,
        "token_weight_versions": action_token_weight_versions,
    }


class AgentExecutorBase(ABC):
    @abstractmethod
    async def execute(self, prompt, label, sampling_params, max_length: int, hf_tokenizer, llm_engine, images=None):
        raise NotImplementedError("AgentExecutorBase.execute is not implemented")


class AgentInstanceBase(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    async def reset(self, states: dict, **kwargs):
        return states

    @abstractmethod
    async def step(self, state_dict: dict, **kwargs):
        raise NotImplementedError("AgentInstance.step is not implemented")


class MultiTurnAgentExecutor(AgentExecutorBase):
    def __init__(self, agent_instance_cls):
        assert issubclass(agent_instance_cls, AgentInstanceBase), "AgentInstance must inherit from AgentInstanceBase"
        self.agent_instance_cls = agent_instance_cls

    async def execute(self, prompt, label, sampling_params, max_length: int, hf_tokenizer, llm_engine, images=None):
        # Treat each AgentInstance as an isolated environment; bind every prompt to its own independent instance
        agent_instance = self.agent_instance_cls()

        # Initialize with reset function
        initial_states = {"observation": prompt, "label": label}
        reset_result = await agent_instance.reset(initial_states)
        observation_text = reset_result["observation"]

        # Tokenize the initial observation
        current_obs_tokens = hf_tokenizer(observation_text, add_special_tokens=False, return_tensors="pt")[
            "input_ids"
        ][0].tolist()

        # Truncate initial observation if it's too long to leave room for generation
        min_generation_tokens = sampling_params.max_tokens if hasattr(sampling_params, "max_tokens") else 1
        max_initial_length = max_length - min_generation_tokens
        if len(current_obs_tokens) > max_initial_length:
            logger.warning(
                f"Initial observation length ({len(current_obs_tokens)}) exceeds max_initial_length ({max_initial_length}). "
                f"Truncating to fit within max_length ({max_length})."
            )
            current_obs_tokens = current_obs_tokens[-max_initial_length:]
            # Also update observation_text to match truncated tokens
            observation_text = hf_tokenizer.decode(current_obs_tokens, skip_special_tokens=False)

        # Initialize tracking variables
        action_ranges = []
        total_reward = 0
        final_scores = 0
        extra_logs = {}
        action_token_weight_versions = []

        if sampling_params.logprobs is not None:
            rollout_log_probs = [0.0] * len(current_obs_tokens)
        else:
            rollout_log_probs = None

        # Execute multiple steps of interaction
        while True:
            # Next sampling budget
            sampling_params.max_tokens = max_length - len(current_obs_tokens)
            # No budget to generate, break
            if sampling_params.max_tokens <= 0:
                break

            generation_output = await _generate_action_with_partial_rollout(
                prompt_token_ids=current_obs_tokens,
                sampling_params=sampling_params,
                max_length=max_length,
                hf_tokenizer=hf_tokenizer,
                llm_engine=llm_engine,
            )
            action_tokens = generation_output["token_ids"]
            action_text = generation_output["text"]
            action_token_weight_versions.extend(generation_output["token_weight_versions"])
            if generation_output["finish_reason"] == "length":
                extra_logs["truncated"] = True

            # Record action range in token space
            action_start = len(current_obs_tokens)
            action_end = action_start + len(action_tokens)
            action_ranges.append((action_start, action_end))

            # Call step function to get environment feedback
            states = {
                "observation_text": observation_text,
                "action_text": action_text,
                "label": label,
                "sampling_params": sampling_params,
            }
            step_result = await agent_instance.step(states)

            total_reward += step_result["rewards"].item()
            final_scores = step_result.get("scores", total_reward)
            environment_feedback_text = step_result["environment_feedback"]
            done = step_result["done"]
            extra_logs = step_result.get("extra_logs", {})

            # Concatenate observation, action, and environment_feedback, then tokenize
            observation_text = observation_text + action_text + environment_feedback_text
            current_obs_tokens = (
                current_obs_tokens
                + action_tokens
                + hf_tokenizer(environment_feedback_text, add_special_tokens=False, return_tensors="pt")["input_ids"][
                    0
                ].tolist()
            )

            # Calculate rollout log probs
            if sampling_params.logprobs is not None:
                rollout_log_probs.extend(generation_output["log_probs"])
                # dummy logprobs for the env feedback tokens
                rollout_log_probs.extend([0.0] * (len(current_obs_tokens) - len(rollout_log_probs)))

            # Get sampling params from the environment step
            if step_result.get("sampling_params", None):
                sampling_params = step_result["sampling_params"]

            if done:
                break

        max_weight_version = (
            max(action_token_weight_versions) if action_token_weight_versions else llm_engine.get_weight_version()
        )
        min_weight_version = min(action_token_weight_versions) if action_token_weight_versions else max_weight_version
        mask_offpolicy = getattr(llm_engine, "mask_offpolicy_in_partial_rollout", False)
        action_loss_mask = [
            0 if mask_offpolicy and version < max_weight_version else 1 for version in action_token_weight_versions
        ]
        partial_old_token_count = sum(version < max_weight_version for version in action_token_weight_versions)

        # Store the final response when agent execution is complete
        final_response = {
            "prompt": prompt,
            "label": label,
            "reward": total_reward,
            "scores": final_scores,
            "observation_tokens": current_obs_tokens,
            "action_ranges": action_ranges,
            "rollout_log_probs": rollout_log_probs,
            "action_loss_mask": action_loss_mask,
            "min_weight_version": min_weight_version,
            "max_weight_version": max_weight_version,
            "partial_old_token_count": partial_old_token_count,
            "extra_logs": extra_logs,
        }
        return final_response


class SingleTurnAgentExecutor(AgentExecutorBase):
    """Single-turn agent executor with optional reward post-processing."""

    def __init__(self, remote_rm_url=None):
        reward_endpoints = [remote_rm_url] if isinstance(remote_rm_url, str) else remote_rm_url
        self.reward_endpoints = reward_endpoints or []

        # Optional user-provided reward_func from a Python file.
        self.reward_func = None
        if self.reward_endpoints and self.reward_endpoints[0].endswith(".py"):
            logger.info(f"Loading custom `reward_func(queries, prompts, labels)` from {self.reward_endpoints[0]}")
            import importlib.util

            spec = importlib.util.spec_from_file_location("reward_func", self.reward_endpoints[0])
            reward_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(reward_module)
            self.reward_func = reward_module.reward_func

    async def execute(self, prompt, label, sampling_params, max_length: int, hf_tokenizer, llm_engine, images=None):
        # Tokenize — for VLM the processor inserts image tokens and returns pixel tensors.
        pil_images = []
        mm_train_inputs = None
        if images and hasattr(hf_tokenizer, "image_processor"):
            from openrlhf.utils.vlm_utils import process_prompt_with_images

            prompt_token_ids, mm_train_inputs, pil_images = process_prompt_with_images(hf_tokenizer, prompt, images)
        else:
            prompt_token_ids = hf_tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"][
                0
            ].tolist()

        # Compute dynamic max_tokens when not explicitly set (prompt + response share max_length budget)
        effective_params = sampling_params
        if sampling_params.max_tokens is None:
            effective_params = deepcopy(sampling_params)
            effective_params.max_tokens = max(1, max_length - len(prompt_token_ids))

        # Truncate prompt if it's too long to leave room for generation
        max_prompt_length = max_length - effective_params.max_tokens
        if len(prompt_token_ids) > max_prompt_length:
            if pil_images:
                raise ValueError(
                    f"VLM prompt length ({len(prompt_token_ids)}) exceeds max_prompt_length ({max_prompt_length}). "
                    f"Truncating VLM prompts would break image token alignment with pixel_values. "
                    f"Please increase --max_len or decrease --max_new_tokens."
                )
            logger.warning(
                f"Prompt length ({len(prompt_token_ids)}) exceeds max_prompt_length ({max_prompt_length}). "
                f"Truncating to fit within max_length ({max_length}) with max_tokens ({effective_params.max_tokens})."
            )
            prompt_token_ids = prompt_token_ids[-max_prompt_length:]

        # Reuse already-loaded PIL images for vLLM generation, including any
        # resumed partial-rollout chunks for the same prompt.
        mm_data = {"image": pil_images} if pil_images else None

        generation_output = await _generate_action_with_partial_rollout(
            prompt_token_ids=prompt_token_ids,
            sampling_params=effective_params,
            max_length=max_length,
            hf_tokenizer=hf_tokenizer,
            llm_engine=llm_engine,
            multi_modal_data=mm_data,
        )
        action_token_ids = generation_output["token_ids"]

        # Check if response was truncated (hit max_tokens length limit)
        is_truncated = generation_output["finish_reason"] == "length"
        action_token_weight_versions = generation_output["token_weight_versions"]
        max_weight_version = (
            max(action_token_weight_versions) if action_token_weight_versions else llm_engine.get_weight_version()
        )
        min_weight_version = min(action_token_weight_versions) if action_token_weight_versions else max_weight_version
        mask_offpolicy = getattr(llm_engine, "mask_offpolicy_in_partial_rollout", False)
        action_loss_mask = [
            0 if mask_offpolicy and version < max_weight_version else 1 for version in action_token_weight_versions
        ]
        partial_old_token_count = sum(version < max_weight_version for version in action_token_weight_versions)

        # Stitch prompt + action together for downstream consumers.
        observation_token_ids = prompt_token_ids + action_token_ids
        action_ranges = [(len(prompt_token_ids), len(observation_token_ids))]

        # Calculate rollout log probs.
        rollout_log_probs = None
        if sampling_params.logprobs is not None and generation_output["log_probs"] is not None:
            rollout_log_probs = [0.0] * len(prompt_token_ids)
            rollout_log_probs.extend(generation_output["log_probs"])

        # Store the final response.
        output = {
            # Original prompt/label/images are echoed for convenience.
            "prompt": prompt,
            "label": label,
            "images": images,
            "mm_train_inputs": mm_train_inputs,
            # Token/text observations and action span.
            "observation_tokens": observation_token_ids,
            "action_ranges": action_ranges,
            "rollout_log_probs": rollout_log_probs,
            "action_loss_mask": action_loss_mask,
            # Truncation flag (finish_reason == "length")
            "truncated": is_truncated,
            "min_weight_version": min_weight_version,
            "max_weight_version": max_weight_version,
            "partial_old_token_count": partial_old_token_count,
            # Reward-related fields (filled by reward/agent variants).
            "reward": None,
            "scores": None,
            "extra_logs": {},
        }

        # Compute reward/score after generation.
        if self.reward_endpoints:
            try:
                query = hf_tokenizer.decode(output["observation_tokens"], skip_special_tokens=False)
                if self.reward_func:
                    rewards_info_list = await self._fetch_rewards_via_func([query], [prompt], [label])
                else:
                    rewards_info_list = await self._fetch_rewards_via_http([query], [prompt], [label])
                rewards_info = rewards_info_list[0] if rewards_info_list else None
                if rewards_info:
                    output.update(
                        reward=rewards_info.get("rewards"),
                        scores=rewards_info.get("scores") or rewards_info.get("rewards"),
                        extra_logs=rewards_info.get("extra_logs") or {},
                    )
            except Exception as e:
                logger.info(f"[SingleTurnExecutor] Failed to fetch reward from remote RM: {e}")

        return output

    async def _fetch_rewards_via_func(self, queries_list, prompts_list, labels_list):
        """Compute rewards via user-provided Python function (thread offload)."""
        batch_size = 1
        num_chunks = (len(queries_list) + batch_size - 1) // batch_size

        tasks = []
        for i in range(num_chunks):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(queries_list))
            tasks.append(
                asyncio.to_thread(
                    self.reward_func,
                    queries_list[start_idx:end_idx],
                    prompts_list[start_idx:end_idx],
                    labels_list[start_idx:end_idx],
                )
            )
        return await asyncio.gather(*tasks)

    async def _fetch_rewards_via_http(self, queries_list, prompts_list, labels_list):
        """HTTP fallback: shard requests across servers."""
        num_servers = len(self.reward_endpoints)
        batch_size = (len(queries_list) + num_servers - 1) // num_servers
        timeout = aiohttp.ClientTimeout(total=180)

        tasks = []
        for i, rm in enumerate(self.reward_endpoints):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(queries_list))
            payload = {
                "query": queries_list[start_idx:end_idx],
                "prompts": prompts_list[start_idx:end_idx],
                "labels": labels_list[start_idx:end_idx],
            }

            async def _post_request(url, data, try_max_times=5):
                for _ in range(try_max_times):
                    try:
                        async with aiohttp.ClientSession(timeout=timeout) as session:
                            async with session.post(url, json=data) as response:
                                response.raise_for_status()
                                return await response.json()
                    except aiohttp.ClientError as e:
                        logger.info(f"Request error, please check: {e}")
                    except Exception as e:  # pragma: no cover - defensive
                        logger.info(f"Unexpected error, please check: {e}")
                    await asyncio.sleep(1)

                raise RuntimeError(
                    f"Request error for {try_max_times} times, returning None. Please check the API server."
                )

            tasks.append(asyncio.create_task(_post_request(rm, payload)))
        return await asyncio.gather(*tasks)
