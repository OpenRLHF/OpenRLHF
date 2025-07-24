import asyncio
import os
from abc import ABC, abstractmethod
import ray
from vllm.inputs import TokensPrompt


class AgentInstanceBase(ABC):
    @abstractmethod
    async def __init__(self, *args, **kwargs):
        pass

    async def reset(self, states: dict, **kwargs):
        return states

    @abstractmethod
    async def step(self, state_dict: dict, **kwargs):
        raise NotImplementedError("AgentInstance.step is not implemented")


class AgentExecutorBase(ABC):
    def __init__(self, agent_instance_cls, max_steps, max_length, llm_engine, hf_tokenizer, result_queue):
        self.llm_engine = llm_engine
        self.hf_tokenizer = hf_tokenizer
        self.max_steps = max_steps
        self.max_length = max_length
        self.result_queue = result_queue
        assert issubclass(agent_instance_cls, AgentInstanceBase), "AgentInstance must inherit from AgentInstanceBase"
        self.agent_instance_cls = ray.remote(agent_instance_cls)

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

    async def execute(self, prompt, label, sampling_params):
        async with self.semaphore:
            # Create a unique agent instance for this prompt with tokenizer
            agent_instance = self.agent_instance_cls.remote()

            # Initialize with reset function
            initial_states = {"observation": prompt, "label": label}
            reset_result = await agent_instance.reset.remote(initial_states)
            observation_text = reset_result["observation"]

            # Tokenize the initial observation
            current_obs_tokens = self.hf_tokenizer(observation_text, add_special_tokens=False, return_tensors="pt")[
                "input_ids"
            ][0].tolist()

            # Initialize tracking variables
            action_ranges = []
            total_reward = 0
            final_scores = 0

            # Execute multiple steps of interaction
            for step_idx in range(self.max_steps):
                # Next sampling budget
                sampling_params.max_tokens = self.max_length - len(current_obs_tokens)
                # No budget to generate, break
                if sampling_params.max_tokens <= 0:
                    break

                # Generate response asynchronously (input and output are token ids)
                request_output = await self.generate(current_obs_tokens, sampling_params)
                action_tokens = request_output.outputs[0].token_ids
                action_text = request_output.outputs[0].text

                # Record action range in token space
                action_start = len(current_obs_tokens)
                action_end = action_start + len(action_tokens)
                action_ranges.append((action_start, action_end))

                # Call step function to get environment feedback
                states = {
                    "observation_text": observation_text,
                    "action_text": action_text,
                    "label": label,
                    "sampling_params": sampling_params,
                }
                step_result = await agent_instance.step.remote(states)

                total_reward += step_result["rewards"].item()
                final_scores = step_result.get("scores", total_reward)
                environment_feedback_text = step_result["environment_feedback"]
                done = step_result["done"]
                extra_logs = step_result.get("extra_logs", {})

                # Concatenate observation, action, and environment_feedback, then tokenize
                observation_text = observation_text + action_text + environment_feedback_text
                current_obs_tokens = (
                    current_obs_tokens
                    + action_tokens
                    + self.hf_tokenizer(environment_feedback_text, add_special_tokens=False, return_tensors="pt")[
                        "input_ids"
                    ][0].tolist()
                )

                # Get sampling params from the environment step
                if step_result.get("sampling_params", None):
                    sampling_params = step_result["sampling_params"]

                if done:
                    break

            ray.kill(agent_instance)

            # Store the final response when agent execution is complete
            final_response = {
                "prompt": prompt,
                "label": label,
                "observation_tokens": current_obs_tokens,
                "reward": total_reward,
                "scores": final_scores,
                "extra_logs": extra_logs,
                "action_ranges": action_ranges,
            }
            await self.result_queue.put(final_response)
