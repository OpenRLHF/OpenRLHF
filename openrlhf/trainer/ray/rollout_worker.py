import asyncio
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Optional

import ray

from openrlhf.utils.agent import AgentExecutorBase
from openrlhf.utils.remote_rm_utils import RemoteRewardModel


class RolloutExecutorBase:
    """Base executor for dispatching generation or agent requests."""

    @staticmethod
    async def _run_with_semaphore(
        coro_fn: Callable[[], Awaitable[Any]], semaphore: Optional[asyncio.Semaphore] = None
    ):
        """Optionally guard a coroutine with a semaphore."""
        if semaphore is not None:
            async with semaphore:
                return await coro_fn()
        return await coro_fn()

    async def execute_with_semaphore(
        self,
        prompt: str,
        label: str,
        sampling_params,
        max_length: int,
        hf_tokenizer,
        request_id: Optional[str],
        llm_engine,
        semaphore: Optional[asyncio.Semaphore] = None,
    ):
        async def _run():
            return await self.execute(
                prompt=prompt,
                label=label,
                sampling_params=sampling_params,
                max_length=max_length,
                hf_tokenizer=hf_tokenizer,
                request_id=request_id,
                llm_engine=llm_engine,
            )

        return await self._run_with_semaphore(_run, semaphore)

    async def execute(
        self,
        prompt: str,
        label: str,
        sampling_params,
        max_length: int,
        hf_tokenizer,
        request_id: Optional[str],
        llm_engine,
    ):
        raise NotImplementedError("Subclasses must implement execute.")


class GenerationExecutor(RolloutExecutorBase):
    """Plain vLLM generation executor (no remote reward)."""

    async def execute(
        self,
        prompt: str,
        label: str,
        sampling_params,
        max_length: int,
        hf_tokenizer,
        request_id: Optional[str],
        llm_engine,
    ):
        output = await self._generate_with_engine(
            prompt=prompt,
            sampling_params=deepcopy(sampling_params),
            max_length=max_length,
            hf_tokenizer=hf_tokenizer,
            request_id=request_id,
            llm_engine=llm_engine,
        )

        output.update(
            {
                "prompt": prompt,
                "label": label,
                "reward": None,
                "scores": None,
                "extra_logs": {},
            }
        )
        return output

    async def _generate_with_engine(self, prompt, sampling_params, max_length, hf_tokenizer, request_id, llm_engine):
        # Tokenize the initial observation.
        prompt_token_ids = hf_tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"][0].tolist()

        # No budget to generate, assert.
        assert max_length > len(prompt_token_ids)

        # Generate response asynchronously (input and output are token ids).
        request_output = ray.get(llm_engine.generate.remote(prompt_token_ids, deepcopy(sampling_params)))

        # Record action range in token space.
        action_tokens = request_output.outputs[0].token_ids
        observation_tokens = prompt_token_ids + action_tokens
        action_ranges = [(len(prompt_token_ids), len(observation_tokens))]

        # Calculate rollout log probs.
        rollout_log_probs = None
        if sampling_params.logprobs is not None and request_output.outputs[0].logprobs is not None:
            rollout_log_probs = [0.0] * len(prompt_token_ids)
            for token_id, logprob_dict in zip(action_tokens, request_output.outputs[0].logprobs):
                token_logprob = logprob_dict.get(token_id)
                rollout_log_probs.append(token_logprob.logprob if token_logprob is not None else 0.0)

        # Store the final response.
        return {
            "observation_tokens": observation_tokens,
            "action_ranges": action_ranges,
            "rollout_log_probs": rollout_log_probs,
            "request_id": request_id,
        }


class RewardedGenerationExecutor(GenerationExecutor):
    """vLLM generation executor with remote reward model post-processing."""

    def __init__(self, remote_rm_url, remote_rm_batch_size: Optional[int] = None):
        rm_args = SimpleNamespace(micro_rollout_batch_size=remote_rm_batch_size or 1)
        self.remote_reward_model = RemoteRewardModel(rm_args, remote_rm_url)

    async def execute(
        self,
        prompt: str,
        label: str,
        sampling_params,
        max_length: int,
        hf_tokenizer,
        request_id: Optional[str],
        llm_engine,
    ):
        output = await super().execute(
            prompt=prompt,
            label=label,
            sampling_params=sampling_params,
            max_length=max_length,
            hf_tokenizer=hf_tokenizer,
            request_id=request_id,
            llm_engine=llm_engine,
        )

        # Compute reward/score after generation.
        reward_val, score_val, extra_logs = await self._compute_remote_reward(
            observation_tokens=output["observation_tokens"],
            prompt=prompt,
            label=label,
            hf_tokenizer=hf_tokenizer,
        )
        output.update(
            {
                "reward": reward_val,
                "scores": score_val,
                "extra_logs": extra_logs,
            }
        )
        return output

    async def _compute_remote_reward(self, observation_tokens, prompt, label, hf_tokenizer):
        """Fetch reward/score from remote RM."""
        try:
            query = hf_tokenizer.decode(observation_tokens, skip_special_tokens=False)
            rewards_info_list = await self.remote_reward_model.get_rewards([query], [prompt], [label])
            if not rewards_info_list:
                return None, None, {}

            rewards_info = rewards_info_list[0]
            reward_val = rewards_info.get("rewards")
            score_val = rewards_info.get("scores") or reward_val
            extra_logs = rewards_info.get("extra_logs") or {}
            return reward_val, score_val, extra_logs
        except Exception as e:
            print(f"[RewardedGenerationExecutor] Failed to fetch reward from remote RM: {e}")
            return None, None, {}


class AgentExecutor(RolloutExecutorBase):
    """Executor wrapper around user-provided AgentExecutor implementations."""

    def __init__(self, agent_func_path: str):
        self.agent_executor_cls = self._load_agent_executor_cls(agent_func_path)

    async def execute(
        self,
        prompt: str,
        label: str,
        sampling_params,
        max_length: int,
        hf_tokenizer,
        request_id: Optional[str],
        llm_engine,
    ):
        agent_executor = self.agent_executor_cls(
            max_length=max_length,
            llm_engine=llm_engine,
            hf_tokenizer=hf_tokenizer,
        )
        return await agent_executor.execute(prompt, label, sampling_params, request_id)

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


@ray.remote
class RolloutWorker:
    """CPU-side worker for both Generation and Agent rollouts."""

    def __init__(
        self,
        agent_func_path: Optional[str] = None,
        remote_rm_url: Optional[str] = None,
        remote_rm_batch_size: Optional[int] = 1,
        max_tasks: Optional[int] = None,
    ):
        # Choose executor priority: Agent > Remote RM > Plain generation.
        if agent_func_path:
            self.executor = AgentExecutor(agent_func_path=agent_func_path)
        elif remote_rm_url:
            self.executor = RewardedGenerationExecutor(remote_rm_url, remote_rm_batch_size)
        else:
            self.executor = GenerationExecutor()

        # Create semaphore to control concurrent task execution.
        self.semaphore = asyncio.Semaphore(max_tasks) if max_tasks else None

    async def execute(
        self,
        llm_engine,
        sampling_params,
        prompt: str,
        label: str,
        max_length: int,
        hf_tokenizer,
        request_id: Optional[str],
        num_samples: int = 1,
    ):
        # Fan out requests; semaphore limits concurrency if configured.
        tasks = []
        for i in range(num_samples):
            tasks.append(
                self.executor.execute_with_semaphore(
                    prompt=prompt,
                    label=label,
                    sampling_params=deepcopy(sampling_params),
                    max_length=max_length,
                    hf_tokenizer=hf_tokenizer,
                    llm_engine=llm_engine,
                    request_id=f"{request_id}_{i}" if request_id is not None else None,
                    semaphore=self.semaphore,
                )
            )

        return await asyncio.gather(*tasks)


def create_rollout_workers(
    num_workers: int,
    worker_cpus: int,
    agent_func_path: Optional[str],
    remote_rm_url: Optional[str],
    remote_rm_batch_size: Optional[int],
) -> list:
    """Spin up a pool of rollout workers, one per desired CPU shard."""
    if num_workers <= 0:
        raise ValueError(f"num_workers must be positive, got {num_workers}")

    workers = []
    for _ in range(num_workers):
        worker = RolloutWorker.options(num_cpus=worker_cpus).remote(
            agent_func_path=agent_func_path,
            remote_rm_url=remote_rm_url,
            remote_rm_batch_size=remote_rm_batch_size,
            max_tasks=worker_cpus,
        )
        workers.append(worker)

    return workers
