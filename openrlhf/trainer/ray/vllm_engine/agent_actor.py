import asyncio
from copy import deepcopy

import ray

from openrlhf.utils.agent import AgentExecutorBase

from .base import BaseLLMRayActor


@ray.remote
class AgentLLMRayActorAsync(BaseLLMRayActor):
    """Agent-style streaming actor using vLLM.AsyncLLMEngine."""

    async def __init__(self, *args, bundle_indices: list = None, **kwargs):
        import vllm

        self.agent_func_path = kwargs.pop("agent_func_path")
        super().__init__(*args, bundle_indices=bundle_indices, **kwargs)

        self.result_queue = asyncio.Queue()

        engine_args = vllm.AsyncEngineArgs(*args, **self.kwargs)
        self.llm = vllm.AsyncLLMEngine.from_engine_args(engine_args)
        await self.llm.is_sleeping()

        # Create AgentExecutor instance
        assert self.agent_func_path.endswith(".py"), "Agent path must be a Python file"
        import importlib.util

        spec = importlib.util.spec_from_file_location("agent_module", self.agent_func_path)
        agent_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(agent_module)

        assert hasattr(agent_module, "AgentExecutor"), "Agent module must contain AgentExecutor class"
        self.agent_executor_cls = agent_module.AgentExecutor
        assert issubclass(
            self.agent_executor_cls, AgentExecutorBase
        ), "AgentExecutor must inherit from AgentExecutorBase"

    async def init_process_group(
        self, master_address, master_port, rank_offset, world_size, group_name, backend, use_ray
    ):
        return await self.llm.collective_rpc(
            "init_process_group",
            args=(master_address, master_port, rank_offset, world_size, group_name, backend, use_ray),
        )

    async def update_weight(self, name, dtype, shape, empty_cache=False):
        return await self.llm.collective_rpc("update_weight", args=(name, dtype, shape, empty_cache))

    async def update_weight_cuda_ipc(self, name, dtype, shape, ipc_handles, empty_cache=False):
        return await self.llm.collective_rpc(
            "update_weight_cuda_ipc", args=(name, dtype, shape, ipc_handles, empty_cache)
        )

    async def reset_prefix_cache(self):
        await self.llm.reset_prefix_cache()

    async def sleep(self, level=1):
        await self.llm.sleep(level=level)

    async def wake_up(self):
        await self.llm.wake_up()

    async def add_requests(
        self, sampling_params, prompts, labels, max_length, hf_tokenizer=None, request_group_id=None
    ):
        self.agent_executor = self.agent_executor_cls(
            max_length=max_length,
            llm_engine=self.llm,
            hf_tokenizer=hf_tokenizer,
            result_queue=self.result_queue,
        )

        tasks = []
        for prompt, label in zip(prompts, labels):
            tasks.append(
                self.agent_executor.execute(
                    prompt,
                    label,
                    deepcopy(sampling_params),
                    request_group_id,
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
