import asyncio
import os
from abc import ABC

from vllm.inputs import TokensPrompt


class AgentInstanceBase(ABC):
    @abstractmethod
    def __init__(self, tokenizer):
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

    async def execute(self, prompt, label, sampling_params):
        async with self.semaphore:
            # Create a unique agent instance for this prompt with tokenizer
            agent_instance = self.agent_instance_cls.remote(self.hf_tokenizer)

            # Initialize with reset function
            initial_states = {"prompt": prompt, "label": label}
            reset_result = await agent_instance.reset.remote(initial_states)

            # Initialize tracking variables
            action_ranges = []
            total_reward = 0
            final_scores = 0
            current_obs_tokens = reset_result["observation_tokens"]

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

                # Decode current observation to text for step function
                observation_text = self.hf_tokenizer.decode(current_obs_tokens, skip_special_tokens=True)

                # Call step function to get environment feedback
                states = {
                    "observation_tokens": current_obs_tokens,
                    "action_tokens": action_tokens,
                    "observation_text": observation_text,
                    "action_text": action_text,
                    "label": label,
                }
                kwargs = {"sampling_params": sampling_params}
                environment_feedback = await agent_instance.step.remote(states, **kwargs)

                total_reward += environment_feedback["rewards"].item()
                final_scores = environment_feedback.get("scores", total_reward)
                current_obs_tokens = environment_feedback["next_observation_tokens"]
                done = environment_feedback["done"]
                extra_logs = environment_feedback.get("extra_logs", {})

                # Get sampling params from the environment step
                if environment_feedback.get("sampling_params", None):
                    sampling_params = environment_feedback["sampling_params"]

                if done:
                    break

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
