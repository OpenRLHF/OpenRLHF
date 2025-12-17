"""Rollout helpers shared by Ray vLLM actors and local runners."""

from copy import deepcopy
from types import SimpleNamespace
from typing import Optional

from openrlhf.utils.agent import AgentExecutorBase
from openrlhf.utils.remote_rm_utils import RemoteRewardModel


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
        rm_args = SimpleNamespace(micro_rollout_batch_size=remote_rm_batch_size or 1)
        self.remote_reward_model = RemoteRewardModel(rm_args, remote_rm_url)

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
        reward_val, score_val, extra_logs = None, None, {}
        try:
            rewards_info_list = await self.remote_reward_model.get_rewards(
                [output["observation_text"]], [prompt], [label]
            )
            if rewards_info_list:
                rewards_info = rewards_info_list[0]
                reward_val = rewards_info.get("rewards")
                score_val = rewards_info.get("scores") or reward_val
                extra_logs = rewards_info.get("extra_logs") or {}
        except Exception as e:
            print(f"[RewardedRolloutWorker] Failed to fetch reward from remote RM: {e}")

        output.update(
            {
                "reward": reward_val,
                "scores": score_val,
                "extra_logs": extra_logs,
            }
        )
        return output


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
