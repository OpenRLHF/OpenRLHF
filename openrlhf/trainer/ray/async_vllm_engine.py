import asyncio
import importlib
import os

import ray

from .vllm_engine import LLMRayActor, logger


@ray.remote
class AsyncLLMRayActor(LLMRayActor):
    def __init__(self, *args, **kwargs):
        tool_dir = kwargs.pop("tool_dir")
        self.tools = {}
        for tool_name in os.listdir(tool_dir):
            if tool_name.endswith(".py"):
                tool_name = tool_name[:-3]
                module = importlib.import_module(f"{tool_dir}.{tool_name}")
                if hasattr(module, "step"):
                    logger.info(f"Importing tool {tool_name} from {tool_dir}")
                    self.tools[tool_name] = getattr(module, "step")

        super().__init__(*args, **kwargs)

    def _init_vllm_engine(self, *args, **kwargs):
        import vllm

        engine_args = vllm.AsyncEngineArgs(*args, **kwargs)
        self.llm = vllm.AsyncLLMEngine.from_engine_args(engine_args)

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
        await self.llm.engine.reset_prefix_cache()

    async def sleep(self, level=1):
        await self.llm.sleep(level=level)

    async def wake_up(self):
        await self.llm.wake_up()

    async def add_requests_tool(self, sampling_params, prompts, tool_names, max_steps=1000):
        """
        Process requests from rank0 and generate responses with multiple agent interactions.
        Each prompt will go through multiple steps of interaction using the step function.

        Args:
            sampling_params: Parameters for sampling
            prompts: List of initial prompts
            tools_types: List of tool types for each prompt
            max_steps: Maximum number of interaction steps for each prompt
        """
        from vllm.utils import random_uuid

        async def generate_async_func(prompts, sampling_params):
            request_id = random_uuid()
            results_generator = self.llm.generate(
                prompts=prompts, sampling_params=sampling_params, request_id=request_id
            )
            async for request_output in results_generator:
                final_output = request_output.outputs
            return final_output

        async def execute_tool(prompt, tool_name):
            # Initialize states and actions for the current prompt
            step = self.tools[tool_name]
            state = prompt
            total_reward = 0

            # Execute multiple steps of interaction
            for step_idx in range(max_steps):
                # Execute tool asynchronously
                action = generate_async_func(prompt, sampling_params)[0]

                # Call step function to get reward and next state
                reward, state, done, extra_info = step(state, action)
                total_reward += reward.item()

                if done:
                    break

            # Store the final response with interaction history
            final_response = {
                "prompt": prompt,
                "states": state,
                "reward": total_reward,
            }
            self.response_queues.put(final_response)

        # Create tasks for all tool executions
        tasks = [execute_tool(prompt, tool_name) for prompt, tool_name in zip(prompts, tool_names)]

        # Wait for all tasks to complete
        await asyncio.gather(*tasks)

    def get_responses(self):
        """
        Return the responses for the actor with the given rank
        """
        return self.response_queues.get()
