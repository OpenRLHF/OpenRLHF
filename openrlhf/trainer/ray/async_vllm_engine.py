import asyncio

import ray

from .vllm_engine import LLMRayActor


@ray.remote
class AsyncLLMRayActor(LLMRayActor):
    def __init__(self, *args, **kwargs):
        # Load agent for step function
        agent_path = kwargs.pop("agent_path")
        self.agent_step = None
        if agent_path.endswith(".py"):
            print(f"Loading agent from {agent_path}")
            import importlib.util

            spec = importlib.util.spec_from_file_location("step", agent_path)
            agent_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(agent_module)
            self.agent_step = ray.remote(agent_module.step)

        # Initialize result queue for streaming completed results
        self.result_queue = asyncio.Queue()
        self.tasks = []

        # Initialize super class
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

    async def add_requests(self, sampling_params, prompts, labels, max_steps=10000):
        """
        Process requests from rank0 and generate responses with multiple agent interactions.
        Each prompt will go through multiple steps of interaction using the step function.
        Results are streamed back as each agent completes its execution.
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

        async def execute_agent(prompt, label):
            # Initialize states and actions for the current prompt
            state = prompt
            action_ranges = []
            total_reward = 0

            # Execute multiple steps of interaction
            for step_idx in range(max_steps):
                # Execute agent asynchronously
                action = await generate_async_func(prompt, sampling_params)[0]

                # Call step function to get reward and next state
                action_ranges.append((len(state), len(state) + len(action)))
                reward, state, done, extra_info = await self.agent_step.remote(state, action, label)
                total_reward += reward.item()

                if done:
                    break

            # Store the final response when agent execution is complete
            final_response = {"prompt": prompt, "state": state, "reward": total_reward, "action_ranges": action_ranges}
            await self.result_queue.put(final_response)

        # Create and start tasks for all agent executions
        for prompt, label in zip(prompts, labels):
            task = asyncio.create_task(execute_agent(prompt, label))
            self.tasks.append(task)

    async def get_responses(self):
        """
        Synchronously get all completed agent results from the queue.
        Waits for all tasks to complete before returning results.
        Returns: List of all completed agent results.
        """
        # Wait for all tasks to complete
        await asyncio.gather(*self.tasks)

        # Get all results from the queue
        results = []
        while not self.result_queue.empty():
            try:
                results.append(self.result_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return results

    async def get_responses_async(self):
        """
        Asynchronously get the next completed agent result from the queue.
        Returns: The next available completed agent result, or None if no result is available within timeout.
        """
        try:
            return await asyncio.wait_for(self.result_queue.get(), timeout=0.5)
        except asyncio.TimeoutError:
            return None
