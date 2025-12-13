import asyncio
import os
from copy import deepcopy
from types import SimpleNamespace

import ray

from openrlhf.utils.remote_rm_utils import RemoteRewardModel

from .base import BaseLLMRayActor


@ray.remote
class LLMRayActorAsync(BaseLLMRayActor):
    """Default streaming actor using AsyncLLMEngine (non-agent)."""

    async def __init__(self, *args, bundle_indices: list = None, **kwargs):
        import vllm

        self.remote_rm_url = kwargs.pop("remote_rm_url", None)
        self.remote_rm_batch_size = kwargs.pop("remote_rm_batch_size", None)
        super().__init__(*args, bundle_indices=bundle_indices, **kwargs)

        self.result_queue = asyncio.Queue()
        self.semaphore = asyncio.Semaphore(int(os.environ.get("OPENRLHF_ASYNC_NUM_TASKS", 128)))

        self.remote_reward_model = None
        if self.remote_rm_url:
            rm_args = SimpleNamespace(micro_rollout_batch_size=self.remote_rm_batch_size or 1)
            self.remote_reward_model = RemoteRewardModel.options(num_cpus=0).remote(rm_args, self.remote_rm_url)

        engine_args = vllm.AsyncEngineArgs(*args, **self.kwargs)
        self.llm = vllm.AsyncLLMEngine.from_engine_args(engine_args)
        await self.llm.is_sleeping()

    async def init_process_group(
        self, master_address, master_port, rank_offset, world_size, group_name, backend, use_ray
    ):
        return await self.llm.collective_rpc(
            "init_process_group",
            args=(master_address, master_port, rank_offset, world_size, group_name, backend, use_ray),
        )

    async def update_weight(self, name, dtype, shape, empty_cache=False):
        return await self.llm.collective_rpc(
            "update_weight",
            args=(name, dtype, shape, empty_cache),
        )

    async def update_weight_cuda_ipc(self, name, dtype, shape, ipc_handles, empty_cache=False):
        return await self.llm.collective_rpc(
            "update_weight_cuda_ipc",
            args=(name, dtype, shape, ipc_handles, empty_cache),
        )

    async def reset_prefix_cache(self):
        await self.llm.reset_prefix_cache()

    async def sleep(self, level=1):
        await self.llm.sleep(level=level)

    async def wake_up(self):
        await self.llm.wake_up()

    async def _generate_single(
        self, prompt, label, sampling_params, max_length, hf_tokenizer=None, request_group_id=None
    ):
        from vllm.inputs import TokensPrompt
        from vllm.utils import random_uuid

        async with self.semaphore:
            # Tokenize the initial prompt
            prompt_token_ids = hf_tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"][
                0
            ].tolist()

            remaining_budget = max_length - len(prompt_token_ids)
            if remaining_budget <= 0:
                await self.result_queue.put(
                    {
                        "prompt": prompt,
                        "label": label,
                        "observation_tokens": prompt_token_ids,
                        "reward": None,
                        "scores": None,
                        "extra_logs": {},
                        "action_ranges": [],
                        "rollout_log_probs": None,
                        "request_group_id": request_group_id,
                    }
                )
                return

            if sampling_params.max_tokens is not None:
                sampling_params.max_tokens = max(1, min(sampling_params.max_tokens, remaining_budget))

            # Generate response asynchronously (input and output are token ids)
            request_id = random_uuid()
            generator = self.llm.generate(
                TokensPrompt(prompt_token_ids=prompt_token_ids),
                sampling_params,
                request_id=request_id,
            )

            final_output = None
            async for request_output in generator:
                final_output = request_output

            output = final_output.outputs[0]
            action_tokens = output.token_ids
            observation_tokens = prompt_token_ids + action_tokens
            action_ranges = [(len(prompt_token_ids), len(observation_tokens))]

            # Calculate rollout log probs
            rollout_log_probs = None
            if sampling_params.logprobs is not None and output.logprobs is not None:
                rollout_log_probs = [0.0] * len(prompt_token_ids)
                for token_id, logprob_dict in zip(action_tokens, output.logprobs):
                    token_logprob = logprob_dict.get(token_id)
                    rollout_log_probs.append(token_logprob.logprob if token_logprob is not None else 0.0)

            reward_val, score_val, extra_logs = await self._maybe_compute_remote_reward(
                observation_tokens, prompt, label, hf_tokenizer
            )

            # Store the final response when agent execution is complete
            final_response = {
                "prompt": prompt,
                "label": label,
                "observation_tokens": observation_tokens,
                "action_ranges": action_ranges,
                "rollout_log_probs": rollout_log_probs,
                "reward": reward_val,
                "scores": score_val,
                "extra_logs": extra_logs,
                "request_group_id": request_group_id,
            }
            await self.result_queue.put(final_response)

    async def _maybe_compute_remote_reward(self, observation_tokens, prompt, label, hf_tokenizer):
        """Fetch reward/score from remote RM if configured."""
        if self.remote_reward_model is None or hf_tokenizer is None:
            return None, None, {}

        try:
            query = hf_tokenizer.decode(observation_tokens, skip_special_tokens=False)
            ref = self.remote_reward_model.get_rewards.remote([query], [prompt], [label])
            rewards_info_list = await asyncio.to_thread(ray.get, ref)
            if not rewards_info_list:
                return None, None, {}

            rewards_info = rewards_info_list[0]
            reward_val = self._extract_scalar(rewards_info.get("rewards"))
            score_val = self._extract_scalar(rewards_info.get("scores")) or reward_val
            extra_logs_raw = rewards_info.get("extra_logs") or {}
            extra_logs = {k: self._extract_scalar(v) for k, v in extra_logs_raw.items()}
            return reward_val, score_val, extra_logs
        except Exception as e:
            print(f"[LLMRayActorAsync] Failed to fetch reward from remote RM: {e}")
            return None, None, {}

    @staticmethod
    def _extract_scalar(value):
        """Convert tensors/lists to a plain scalar value."""
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return LLMRayActorAsync._extract_scalar(value[0]) if value else None
        try:
            return float(value)
        except (TypeError, ValueError):
            pass

        # Handle objects with .item(), such as torch/numpy tensors
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                return None
        return None

    async def add_requests(
        self, sampling_params, prompts, labels, max_length, hf_tokenizer=None, request_group_id=None
    ):
        tasks = []
        for prompt, label in zip(prompts, labels):
            tasks.append(
                self._generate_single(
                    prompt,
                    label,
                    deepcopy(sampling_params),
                    max_length,
                    hf_tokenizer=hf_tokenizer,
                    request_group_id=request_group_id,
                )
            )

        await asyncio.gather(*tasks)

    async def get_responses(self, request_group_id=None):
        if request_group_id is None:
            results = []
            while not self.result_queue.empty():
                results.append(await self.result_queue.get())
            return results

        matching_results = []
        unmatched = []

        while True:
            try:
                item = self.result_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            if item.get("request_group_id") == request_group_id:
                matching_results.append(item)
            else:
                unmatched.append(item)

        for item in unmatched:
            self.result_queue.put_nowait(item)

        return matching_results
