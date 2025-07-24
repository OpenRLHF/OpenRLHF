import asyncio
import os
from abc import ABC

from vllm.inputs import TokensPrompt


class AgentInstanceBase(ABC):
    @abstractmethod
    def __init__(self):
        raise NotImplementedError("AgentInstance.__init__ is not implemented")

    async def reset(self, states: dict, **kwargs):
        return states

    @abstractmethod
    async def step(self, state_dict: dict):
        raise NotImplementedError("AgentInstance.step is not implemented")


class AgentExecutorBase(ABC):
    def __init__(self, agent_instance_cls, max_steps, max_length, llm_engine, result_queue):
        self.llm_engine = llm_engine
        self.hf_tokenizer = llm_engine.tokenizer
        self.max_steps = max_steps
        self.max_length = max_length
        self.result_queue = result_queue
        self.agent_instance_cls = agent_instance_cls

        # Create semaphore to control concurrent task execution
        NUM_TASKS = os.environ.get("OPENRLHF_ASYNC_NUM_TASKS", 128)
        self.semaphore = asyncio.Semaphore(NUM_TASKS)

    async def generate(self, prompt_ids, sampling_params):
        from vllm.utils import random_uuid

        prompts = TokensPrompt(prompt_token_ids=prompt_ids)
        request_id = random_uuid()
        results_generator = self.llm_engine.generate(prompts, sampling_params, request_id)
        final_output = None
        async for request_output in results_generator:
            final_output = request_output
        return final_output

    async def execute_agent(self, prompt, label, sampling_params):
        async with self.semaphore:
            # Create a unique agent instance for this prompt
            agent_instance = self.agent_instance_cls.remote()

            # Initialize observations and actions for the current prompt
            observation_tokens = self.hf_tokenizer(observation, add_special_tokens=False, return_tensors="pt")[
                "input_ids"
            ][0]
            action_ranges = []
            total_reward = 0
            final_scores = 0

            # Execute multiple steps of interaction
            for step_idx in range(self.max_steps):
                # Next sampling budget
                sampling_params.max_tokens = self.max_length - len(observation_tokens)
                # No budget to generate, break
                if sampling_params.max_tokens <= 0:
                    break

                # Generate response asynchronously
                request_output = await self.generate(observation_tokens, sampling_params)
                action = request_output.outputs[0].text
                action_ranges.append((len(observation), len(observation) + len(action)))

                # Call step function to get reward and next observation
                # Use asyncio.to_thread to make Ray remote call non-blocking
                kwargs = {"sampling_params": sampling_params}
                result = await agent_instance.step.remote(observation, action, label, **kwargs)
                total_reward += result["rewards"].item()
                final_scores = result.get("scores", total_reward)
                observation = result["next_observation"]
                done = result["done"]
                extra_logs = result.get("extra_logs", {})

                # Get sampling params from the environment step
                if result.get("sampling_params", None):
                    sampling_params = result["sampling_params"]

                if done:
                    break

            # Store the final response when agent execution is complete
            final_response = {
                "prompt": prompt,
                "label": label,
                "observation": observation,
                "reward": total_reward,
                "scores": final_scores,
                "extra_logs": extra_logs,
                "action_ranges": action_ranges,
            }
            await self.result_queue.put(final_response)
