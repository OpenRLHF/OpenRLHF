import random
from typing import Any, Dict

import torch
from openrlhf.utils.agent import AgentExecutorBase, AgentInstanceBase


# A simple n-step random environment
class AgentInstance(AgentInstanceBase):
    async def __init__(self, *args, **kwargs):
        self.step_idx = 0
        self.max_steps = random.randint(1, 3)  # 1-3 steps

    async def reset(self, states: dict, **kwargs):
        """Initialize the environment and return initial observation

        Args:
            states: Dictionary containing prompt and label

        Returns:
            str: Initial observation text
        """
        return {"observation": states["observation"]}  # Return original text observation

    async def step(self, states: dict, **kwargs) -> Dict[str, Any]:
        """Execute one step of verification and return environment feedback

        Args:
            states: Dictionary containing observation_text, action_text, and label

        Returns:
            Dict[str, Any]: A dictionary containing:
                - rewards: Reward value for advantage calculation
                - scores: Reward value for dynamic filtering
                - environment_feedback: The environment feedback text
                - done: Boolean indicating if the episode is complete
                - sampling_params: Parameters for vLLM sampling
                - extra_logs: Additional logging information
        """
        print(f"step_idx: {self.step_idx}, max_steps: {self.max_steps}")

        observation_text = states["observation_text"]
        action_text = states["action_text"]
        label = states["label"]

        # Check if episode is done
        done = self.step_idx >= self.max_steps
        reward = torch.randint(0, 2, (1,)).float() if done else torch.tensor(0)

        # Generate environment feedback based on whether episode is done
        environment_feedback = (
            "\n\nHuman: [CORRECT]\n</s>"
            if done
            else "\n\nHuman: [INCORRECT]\nPlease analyze the issues and try again.\n</s>\n\nAssistant: "
        )

        self.step_idx += 1

        return {
            "rewards": reward,  # Rewards for advantage calculation
            "scores": reward,  # Scores for dynamic filtering (0-1 reward)
            "environment_feedback": environment_feedback,  # Environment feedback text
            "done": done,  # Boolean indicating if the episode is complete
            "sampling_params": states.get("sampling_params", None),  # Parameters for vLLM sampling in next step
            "extra_logs": {"dummy_scores": reward},  # Additional logging information
        }


class AgentExecutor(AgentExecutorBase):
    def __init__(self, max_steps, max_length, llm_engine, hf_tokenizer, result_queue):
        super().__init__(AgentInstance, max_steps, max_length, llm_engine, hf_tokenizer, result_queue)

    async def execute(self, prompt, label, sampling_params):
        # You could override the execute function of AgentExecutorBase to add custom agent running logic
        return await super().execute(prompt, label, sampling_params)
