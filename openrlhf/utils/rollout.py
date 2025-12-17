"""Rollout helpers shared by Ray vLLM actors and local runners."""

import asyncio
from copy import deepcopy
from typing import Optional

import aiohttp

from openrlhf.utils.agent import AgentExecutorBase
from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)


class RolloutWorker:
    """Plain vLLM rollout worker (no remote reward)."""

    async def run(self, prompt, label, sampling_params, max_length: int, hf_tokenizer, llm_engine):
        # Tokenize the initial observation.
        prompt_token_ids = hf_tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"][0].tolist()

        # Generate one continuation from the engine.
        request_output = await llm_engine.generate(prompt_token_ids, deepcopy(sampling_params))
        action_token_ids = request_output.outputs[0].token_ids
        action_text = request_output.outputs[0].text

        # Stitch prompt + action together for downstream consumers.
        observation_token_ids = prompt_token_ids + action_token_ids
        observation_text = prompt + action_text
        action_ranges = [(len(prompt_token_ids), len(observation_token_ids))]

        # Calculate rollout log probs.
        rollout_log_probs = None
        if sampling_params.logprobs is not None and request_output.outputs[0].logprobs is not None:
            rollout_log_probs = [0.0] * len(prompt_token_ids)
            for token_id, logprob_dict in zip(action_token_ids, request_output.outputs[0].logprobs):
                token_logprob = logprob_dict.get(token_id)
                rollout_log_probs.append(token_logprob.logprob if token_logprob is not None else 0.0)

        # Store the final response.
        return {
            # Original prompt/label are echoed for convenience.
            "prompt": prompt,
            "label": label,
            # Token/text observations and action span.
            "observation_tokens": observation_token_ids,
            "observation_text": observation_text,
            "action_ranges": action_ranges,
            "rollout_log_probs": rollout_log_probs,
            # Reward-related fields (filled by reward/agent variants).
            "reward": None,
            "scores": None,
            "extra_logs": {},
        }


class RolloutAndRewardWorker(RolloutWorker):
    """vLLM rollout worker with remote reward model post-processing."""

    def __init__(self, remote_rm_url, remote_rm_batch_size: Optional[int] = None):
        self.remote_rm_url = [remote_rm_url] if isinstance(remote_rm_url, str) else remote_rm_url
        self.micro_rollout_batch_size = remote_rm_batch_size or 1

        # Optional user-provided reward_func from a Python file.
        self.custom_reward_func = None
        if self.remote_rm_url and self.remote_rm_url[0].endswith(".py"):
            logger.info(f"Loading custom `reward_func(queries, prompts, labels)` from {self.remote_rm_url[0]}")
            import importlib.util

            spec = importlib.util.spec_from_file_location("reward_func", self.remote_rm_url[0])
            reward_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(reward_module)
            self.custom_reward_func = reward_module.reward_func

    async def run(self, prompt: str, label: str, sampling_params, max_length: int, hf_tokenizer, llm_engine):
        output = await super().run(
            prompt=prompt,
            label=label,
            sampling_params=sampling_params,
            max_length=max_length,
            hf_tokenizer=hf_tokenizer,
            llm_engine=llm_engine,
        )

        # Compute reward/score after generation.
        try:
            if self.custom_reward_func:
                rewards_info_list = await self._get_rewards_via_func([output["observation_text"]], [prompt], [label])
            else:
                rewards_info_list = await self._get_rewards_via_http([output["observation_text"]], [prompt], [label])

            if rewards_info_list:
                rewards_info = rewards_info_list[0]
                output.update(
                    reward=rewards_info.get("rewards"),
                    scores=rewards_info.get("scores") or rewards_info.get("rewards"),
                    extra_logs=rewards_info.get("extra_logs") or {},
                )
        except Exception as e:
            logger.info(f"[RolloutAndRewardWorker] Failed to fetch reward from remote RM: {e}")

        return output

    async def _get_rewards_via_func(self, queries_list, prompts_list, labels_list):
        """Compute rewards via user-provided Python function (thread offload)."""
        batch_size = self.micro_rollout_batch_size
        num_chunks = (len(queries_list) + batch_size - 1) // batch_size

        tasks = []
        for i in range(num_chunks):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(queries_list))
            tasks.append(
                asyncio.to_thread(
                    self.custom_reward_func,
                    queries_list[start_idx:end_idx],
                    prompts_list[start_idx:end_idx],
                    labels_list[start_idx:end_idx],
                )
            )
        return await asyncio.gather(*tasks)

    async def _get_rewards_via_http(self, queries_list, prompts_list, labels_list):
        """HTTP fallback: shard requests across servers."""
        num_servers = len(self.remote_rm_url)
        batch_size = (len(queries_list) + num_servers - 1) // num_servers

        tasks = []
        for i, rm in enumerate(self.remote_rm_url):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(queries_list))
            tasks.append(
                asyncio.create_task(
                    self._request_api_wrapper(
                        rm,
                        {
                            "query": queries_list[start_idx:end_idx],
                            "prompts": prompts_list[start_idx:end_idx],
                            "labels": labels_list[start_idx:end_idx],
                        },
                    )
                )
            )
        return await asyncio.gather(*tasks)

    async def _request_api_wrapper(self, url, data, try_max_times=5):
        """Async HTTP POST with retries using aiohttp."""
        timeout = aiohttp.ClientTimeout(total=180)
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

        raise RuntimeError(f"Request error for {try_max_times} times, returning None. Please check the API server.")


class RolloutWithAgentWorker:
    """Wrapper that delegates rollout control to a user-provided AgentExecutor."""

    def __init__(self, agent_func_path: str):
        self.agent_executor_cls = self._load_agent_executor_cls(agent_func_path)

    async def run(self, prompt, label, sampling_params, max_length: int, hf_tokenizer, llm_engine):
        agent_executor = self.agent_executor_cls(
            max_length=max_length,
            llm_engine=llm_engine,
            hf_tokenizer=hf_tokenizer,
        )
        return await agent_executor.execute(prompt, label, sampling_params)

    def _load_agent_executor_cls(self, agent_func_path: str):
        assert agent_func_path.endswith(".py"), "Agent path must be a Python file"
        import importlib.util

        spec = importlib.util.spec_from_file_location("agent_module", agent_func_path)
        agent_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(agent_module)

        assert hasattr(agent_module, "AgentExecutor"), "Agent module must contain AgentExecutor class"
        agent_executor_cls = agent_module.AgentExecutor
        assert issubclass(agent_executor_cls, AgentExecutorBase), "AgentExecutor must inherit from AgentExecutorBase"
        return agent_executor_cls
