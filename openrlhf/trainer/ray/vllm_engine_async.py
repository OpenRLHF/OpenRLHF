import asyncio
import os

import ray

from openrlhf.utils.agent import AgentExecutorBase

from .vllm_engine import BaseLLMRayActor


@ray.remote
class LLMRayActorAsync(BaseLLMRayActor):
    async def __init__(self, *args, bundle_indices: list = None, **kwargs):
        self.agent_func_path = kwargs.pop("agent_func_path")
        # Initialize super class
        super().__init__(*args, bundle_indices=bundle_indices, **kwargs)

        # Initialize result queue for streaming completed results
        self.result_queue = asyncio.Queue()
        self.agent_executor = None

        os.environ["VLLM_USE_V1"] = "1"
        import vllm
        from packaging import version

        assert version.parse(vllm.__version__) > version.parse("0.8.5"), "Asyn VLLM version must be greater than 0.8.5"

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
        self, sampling_params, prompts, labels, max_length, hf_tokenizer=None, max_steps=10000, request_group_id=None
    ):
        """
        Process requests from rank0 and generate responses with multiple agent interactions.
        Each prompt will go through multiple steps of interaction using the AgentExecutor.
        Results are streamed back as each agent completes its execution.

        Args:
            sampling_params: Parameters for sampling
            prompts: List of prompts to process
            labels: List of labels corresponding to prompts
            max_steps: Maximum number of interaction steps
            request_group_id: Optional ID to group related requests together
        """

        # Create AgentExecutor instance
        if self.agent_executor is None:
            assert self.agent_func_path.endswith(".py"), "Agent path must be a Python file"
            import importlib.util

            spec = importlib.util.spec_from_file_location("agent_module", self.agent_func_path)
            agent_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(agent_module)

            # Load AgentExecutor class instead of step function
            assert hasattr(agent_module, "AgentExecutor"), "Agent module must contain AgentExecutor class"
            self.agent_executor_cls = agent_module.AgentExecutor
            assert issubclass(self.agent_executor_cls, AgentExecutorBase), (
                "AgentExecutor must inherit from AgentExecutorBase"
            )

            self.agent_executor = self.agent_executor_cls(
                max_steps=max_steps,
                max_length=max_length,
                llm_engine=self.llm,
                hf_tokenizer=hf_tokenizer,
                result_queue=self.result_queue,
            )

        # Create and start tasks for all agent executions with controlled concurrency
        import copy

        tasks = []
        for prompt, label in zip(prompts, labels):
            tasks.append(self.agent_executor.execute(prompt, label, copy.deepcopy(sampling_params), request_group_id))

        # Run the async code using the class's event loop
        await asyncio.gather(*tasks)

    async def get_responses(self, request_group_id=None):
        """
        Get completed agent results from the queue.
        If request_group_id is provided, only return results for that group.

        Args:
            request_group_id: Optional ID to filter results by group

        Returns:
            List of agent results (all if request_group_id is None, filtered otherwise)
        """
        if request_group_id is None:
            # Original behavior: return all results
            results = []
            while not self.result_queue.empty():
                results.append(await self.result_queue.get())
            return results
        else:
            # Filter by request_group_id
            matching_results = []
            temp_queue = []

            # Extract all items from queue
            while not self.result_queue.empty():
                item = await self.result_queue.get()
                if item.get("request_group_id") == request_group_id:
                    matching_results.append(item)
                else:
                    temp_queue.append(item)

            # Put back non-matching items
            for item in temp_queue:
                await self.result_queue.put(item)

            return matching_results
