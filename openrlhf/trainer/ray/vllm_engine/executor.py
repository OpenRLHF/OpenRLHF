import asyncio
from types import SimpleNamespace
from typing import Optional

from openrlhf.utils.agent import AgentExecutorBase
from openrlhf.utils.remote_rm_utils import RemoteRewardModel


class BaseRequestExecutor:
    """Base executor for dispatching generation or agent requests."""

    async def execute(
        self,
        prompt,
        label,
        sampling_params,
        max_length,
        hf_tokenizer,
        request_id,
        llm_engine,
        result_queue,
        semaphore: asyncio.Semaphore = None,
    ):
        raise NotImplementedError


class GenerationRequestExecutor(BaseRequestExecutor):
    """Default vLLM generation executor with optional remote reward model support."""

    def __init__(self, remote_rm_url: Optional[str] = None, remote_rm_batch_size: Optional[int] = None):
        self.remote_reward_model = None
        if remote_rm_url:
            rm_args = SimpleNamespace(micro_rollout_batch_size=remote_rm_batch_size or 1)
            self.remote_reward_model = RemoteRewardModel(rm_args, remote_rm_url)

    async def execute(
        self,
        prompt,
        label,
        sampling_params,
        max_length,
        hf_tokenizer,
        request_id,
        llm_engine,
        result_queue,
        semaphore: asyncio.Semaphore = None,
    ):
        from vllm.inputs import TokensPrompt
        from vllm.utils import random_uuid

        async with semaphore:
            if hf_tokenizer is None:
                raise ValueError("hf_tokenizer is required for GenerationRequestExecutor.")

            prompt_token_ids = hf_tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"][
                0
            ].tolist()

            remaining_budget = max_length - len(prompt_token_ids)
            if remaining_budget <= 0:
                await result_queue.put(
                    {
                        "prompt": prompt,
                        "label": label,
                        "observation_tokens": prompt_token_ids,
                        "reward": None,
                        "scores": None,
                        "extra_logs": {},
                        "action_ranges": [],
                        "rollout_log_probs": None,
                        "request_id": request_id,
                    }
                )
                return

            if sampling_params.max_tokens is not None:
                sampling_params.max_tokens = max(1, min(sampling_params.max_tokens, remaining_budget))

            generation_request_id = random_uuid()
            generator = llm_engine.generate(
                TokensPrompt(prompt_token_ids=prompt_token_ids),
                sampling_params,
                request_id=generation_request_id,
            )

            final_output = None
            async for request_output in generator:
                final_output = request_output

            output = final_output.outputs[0]
            action_tokens = output.token_ids
            observation_tokens = prompt_token_ids + action_tokens
            action_ranges = [(len(prompt_token_ids), len(observation_tokens))]

            rollout_log_probs = None
            if sampling_params.logprobs is not None and output.logprobs is not None:
                rollout_log_probs = [0.0] * len(prompt_token_ids)
                for token_id, logprob_dict in zip(action_tokens, output.logprobs):
                    token_logprob = logprob_dict.get(token_id)
                    rollout_log_probs.append(token_logprob.logprob if token_logprob is not None else 0.0)

            reward_val, score_val, extra_logs = await self._maybe_compute_remote_reward(
                observation_tokens, prompt, label, hf_tokenizer
            )

            final_response = {
                "prompt": prompt,
                "label": label,
                "observation_tokens": observation_tokens,
                "action_ranges": action_ranges,
                "rollout_log_probs": rollout_log_probs,
                "reward": reward_val,
                "scores": score_val,
                "extra_logs": extra_logs,
                "request_id": request_id,
            }
            await result_queue.put(final_response)

    async def _maybe_compute_remote_reward(self, observation_tokens, prompt, label, hf_tokenizer):
        """Fetch reward/score from remote RM if configured."""
        if self.remote_reward_model is None or hf_tokenizer is None:
            return None, None, {}

        try:
            query = hf_tokenizer.decode(observation_tokens, skip_special_tokens=False)
            rewards_info_list = await self.remote_reward_model.get_rewards([query], [prompt], [label])
            if not rewards_info_list:
                return None, None, {}

            rewards_info = rewards_info_list[0]
            reward_val = self._extract_scalar(rewards_info.get("rewards"))
            score_val = self._extract_scalar(rewards_info.get("scores")) or reward_val
            extra_logs_raw = rewards_info.get("extra_logs") or {}
            extra_logs = {k: self._extract_scalar(v) for k, v in extra_logs_raw.items()}
            return reward_val, score_val, extra_logs
        except Exception as e:
            print(f"[GenerationRequestExecutor] Failed to fetch reward from remote RM: {e}")
            return None, None, {}

    @staticmethod
    def _extract_scalar(value):
        """Convert tensors/lists to a plain scalar value."""
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return GenerationRequestExecutor._extract_scalar(value[0]) if value else None
        try:
            return float(value)
        except (TypeError, ValueError):
            pass

        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                return None
        return None


class AgentRequestExecutor(BaseRequestExecutor):
    """Executor wrapper around user-provided AgentExecutor implementations."""

    def __init__(self, agent_func_path: str):
        self.agent_func_path = agent_func_path
        self.agent_executor_cls = None
        self.agent_executor = None
        self._cached_max_length = None

    async def execute(
        self,
        prompt,
        label,
        sampling_params,
        max_length,
        hf_tokenizer,
        request_id=None,
        llm_engine=None,
        result_queue=None,
        semaphore: asyncio.Semaphore = None,
    ):
        if hf_tokenizer is None:
            raise ValueError("hf_tokenizer is required for AgentRequestExecutor.")

        # Lazily load executor class after deserialization to avoid pickling issues.
        if self.agent_executor_cls is None:
            self.agent_executor_cls = self._load_agent_executor_cls(self.agent_func_path)

        self._ensure_agent_executor(max_length, hf_tokenizer, llm_engine, result_queue)
        await self.agent_executor.execute(prompt, label, sampling_params, request_id)

    def _ensure_agent_executor(self, max_length, hf_tokenizer, llm_engine, result_queue):
        """Instantiate or refresh the agent executor when settings change."""
        if self.agent_executor is None or self._cached_max_length != max_length:
            self.agent_executor = self.agent_executor_cls(
                max_length=max_length,
                llm_engine=llm_engine,
                hf_tokenizer=hf_tokenizer,
                result_queue=result_queue,
            )
            self._cached_max_length = max_length

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
