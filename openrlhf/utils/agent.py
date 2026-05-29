import asyncio
from abc import ABC, abstractmethod
from copy import deepcopy

import aiohttp

from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)


class AgentExecutorBase(ABC):
    @abstractmethod
    async def execute(self, prompt, label, sampling_params, max_length: int, hf_tokenizer, llm_engine):
        raise NotImplementedError("AgentExecutorBase.execute is not implemented")


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

    async def execute(self, prompt, label, sampling_params, max_length: int, hf_tokenizer, llm_engine):
        prompt_token_ids = hf_tokenizer(text=prompt, add_special_tokens=False, return_tensors="pt")["input_ids"][
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
            logger.warning(
                f"Prompt length ({len(prompt_token_ids)}) exceeds max_prompt_length ({max_prompt_length}). "
                f"Truncating to fit within max_length ({max_length}) with max_tokens ({effective_params.max_tokens})."
            )
            prompt_token_ids = prompt_token_ids[-max_prompt_length:]

        # Generate one continuation from the engine.
        request_output = await llm_engine.generate(prompt_token_ids, deepcopy(effective_params))
        generation_output = request_output.outputs[0]
        action_token_ids = generation_output.token_ids

        # Check if response was truncated (hit max_tokens length limit)
        is_truncated = generation_output.finish_reason == "length"

        # Stitch prompt + action together for downstream consumers.
        observation_token_ids = prompt_token_ids + action_token_ids
        action_ranges = [(len(prompt_token_ids), len(observation_token_ids))]

        # Calculate rollout log probs.
        rollout_log_probs = None
        if sampling_params.logprobs is not None and generation_output.logprobs is not None:
            rollout_log_probs = [0.0] * len(prompt_token_ids)
            for token_id, logprob_dict in zip(action_token_ids, generation_output.logprobs):
                token_logprob = logprob_dict.get(token_id)
                rollout_log_probs.append(token_logprob.logprob if token_logprob is not None else 0.0)

        # Store the final response.
        output = {
            # Original prompt/label are echoed for convenience.
            "prompt": prompt,
            "label": label,
            # Token/text observations and action span.
            "observation_tokens": observation_token_ids,
            "action_ranges": action_ranges,
            "rollout_log_probs": rollout_log_probs,
            # Truncation flag (finish_reason == "length")
            "truncated": is_truncated,
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
                    score_value = rewards_info.get("scores")
                    if score_value is None:
                        score_value = rewards_info.get("rewards")
                    output.update(
                        reward=rewards_info.get("rewards"),
                        scores=score_value,
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
