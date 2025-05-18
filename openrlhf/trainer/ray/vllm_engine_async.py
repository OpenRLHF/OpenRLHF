import asyncio
import os

import ray

from .vllm_engine import BaseLLMRayActor


@ray.remote
class AgentInstance:
    def __init__(self, agent_func_path):
        if agent_func_path.endswith(".py"):
            import importlib.util

            spec = importlib.util.spec_from_file_location("step", agent_func_path)
            agent_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(agent_module)
            self.agent_step = agent_module.step
        else:
            raise ValueError("Agent path must be a Python file")

    async def step(self, state, action, label):
        return await self.agent_step(state, action, label)


@ray.remote
class LLMRayActorAsync(BaseLLMRayActor):
    async def __init__(self, *args, bundle_indices: list = None, **kwargs):
        self.agent_func_path = kwargs.pop("agent_func_path")

        # Initialize super class
        super().__init__(*args, bundle_indices=bundle_indices, **kwargs)

        # Initialize result queue for streaming completed results
        self.result_queue = asyncio.Queue()

        import vllm

        engine_args = vllm.AsyncEngineArgs(*args, **self.kwargs)
        self.llm = vllm.AsyncLLMEngine.from_engine_args(engine_args)
        await self.llm.is_sleeping()

    def init_process_group(self, master_address, master_port, rank_offset, world_size, group_name, backend, use_ray):
        return self.llm.engine.model_executor.collective_rpc(
            "init_process_group",
            args=(master_address, master_port, rank_offset, world_size, group_name, backend, use_ray),
        )

    def update_weight(self, name, dtype, shape, empty_cache=False):
        return self.llm.engine.model_executor.collective_rpc("update_weight", args=(name, dtype, shape, empty_cache))

    def update_weight_cuda_ipc(self, name, dtype, shape, ipc_handles, empty_cache=False):
        return self.llm.engine.model_executor.collective_rpc(
            "update_weight_cuda_ipc", args=(name, dtype, shape, ipc_handles, empty_cache)
        )

    def reset_prefix_cache(self):
        self.llm.engine.reset_prefix_cache()

    async def sleep(self, level=1):
        await self.llm.sleep(level=level)

    async def wake_up(self):
        await self.llm.wake_up()

    async def add_requests(self, sampling_params, prompts, labels, max_length, max_steps=10000):
        """
        Process requests from rank0 and generate responses with multiple agent interactions.
        Each prompt will go through multiple steps of interaction using the step function.
        Results are streamed back as each agent completes its execution.

        Args:
            sampling_params: Parameters for sampling
            prompts: List of prompts to process
            labels: List of labels corresponding to prompts
            max_steps: Maximum number of interaction steps
            micro_forward_batch_size: Number of prompts to process in each concurrent task
        """

        # Create semaphore to control concurrent task execution
        NUM_TASKS = os.environ.get("OPENRLHF_ASYNC_NUM_TASKS", 128)
        semaphore = asyncio.Semaphore(NUM_TASKS)

        async def execute_agent(prompt, label, sampling_params):
            async with semaphore:
                # Create a unique agent instance for this prompt
                agent_instance = AgentInstance.remote(self.agent_func_path)

                # Initialize states and actions for the current prompt
                state = prompt
                action_ranges = []
                total_reward = 0

                # Execute multiple steps of interaction
                for step_idx in range(max_steps):
                    # No budget to generate, break
                    if sampling_params.max_tokens <= 0:
                        break

                    # Generate response asynchronously
                    request_output = await self.generate_async(state, sampling_params)
                    action = request_output.outputs[0].text
                    action_ranges.append((len(state), len(state) + len(action)))
                    # Next sampling budget
                    sampling_params.max_tokens = max_length - (
                        len(request_output.prompt_token_ids) + len(request_output.outputs[0].token_ids)
                    )

                    # Call step function to get reward and next state
                    # Use asyncio.to_thread to make Ray remote call non-blocking
                    result = await agent_instance.step.remote(state, action, label)
                    reward, state, done, extra_info = result
                    total_reward += reward.item()

                    if done:
                        break

                ray.kill(agent_instance)

                # Store the final response when agent execution is complete
                final_response = {
                    "prompt": prompt,
                    "label": label,
                    "state": state,
                    "reward": total_reward,
                    "action_ranges": action_ranges,
                }
                await self.result_queue.put(final_response)

        # Create and start tasks for all agent executions with controlled concurrency
        import copy

        tasks = []
        for prompt, label in zip(prompts, labels):
            tasks.append(execute_agent(prompt, label, copy.deepcopy(sampling_params)))

        # Run the async code using the class's event loop
        await asyncio.gather(*tasks)

    async def generate_async(self, prompts, sampling_params):
        from vllm.utils import random_uuid

        request_id = random_uuid()
        results_generator = self.llm.generate(prompts, sampling_params, request_id)
        final_output = None
        async for request_output in results_generator:
            final_output = request_output
        return final_output

    async def get_responses(self):
        """
        Synchronously get all completed agent results from the queue.
        Waits for all tasks to complete before returning results.
        Returns: List of all completed agent results.
        """
        # Get all results from the queue
        results = []
        while not self.result_queue.empty():
            try:
                results.append(await self.result_queue.get())
            except asyncio.QueueEmpty:
                break
        return results
