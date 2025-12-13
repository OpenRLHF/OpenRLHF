import asyncio
import os
from copy import deepcopy

import ray

from .base import BaseLLMRayActor


@ray.remote
class LLMRayActorAsync(BaseLLMRayActor):
    """Default streaming actor using AsyncLLMEngine (non-agent)."""

    async def __init__(self, *args, bundle_indices: list = None, **kwargs):
        import vllm

        super().__init__(*args, bundle_indices=bundle_indices, **kwargs)

        self.result_queue = asyncio.Queue()
        self.semaphore = asyncio.Semaphore(int(os.environ.get("OPENRLHF_ASYNC_NUM_TASKS", 128)))

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

            params = deepcopy(sampling_params)
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

            if params.max_tokens is not None:
                params.max_tokens = max(1, min(params.max_tokens, remaining_budget))

            request_id = random_uuid()
            generator = self.llm.generate(
                TokensPrompt(prompt_token_ids=prompt_token_ids), params, request_id=request_id
            )

            final_output = None
            async for request_output in generator:
                final_output = request_output

            output = final_output.outputs[0]
            action_tokens = output.token_ids
            observation_tokens = prompt_token_ids + action_tokens
            action_ranges = [(len(prompt_token_ids), len(observation_tokens))]

            rollout_log_probs = None
            if params.logprobs is not None and output.logprobs is not None:
                rollout_log_probs = [0.0] * len(prompt_token_ids)
                for token_id, logprob_dict in zip(action_tokens, output.logprobs):
                    token_logprob = logprob_dict.get(token_id)
                    rollout_log_probs.append(token_logprob.logprob if token_logprob is not None else 0.0)

            await self.result_queue.put(
                {
                    "prompt": prompt,
                    "label": label,
                    "observation_tokens": observation_tokens,
                    "action_ranges": action_ranges,
                    "rollout_log_probs": rollout_log_probs,
                    "reward": None,
                    "scores": None,
                    "extra_logs": {},
                    "request_group_id": request_group_id,
                }
            )

    async def add_requests(
        self, sampling_params, prompts, labels, max_length, hf_tokenizer=None, request_group_id=None
    ):
        tasks = []
        for prompt, label in zip(prompts, labels):
            tasks.append(
                self._generate_single(
                    prompt,
                    label,
                    sampling_params,
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
