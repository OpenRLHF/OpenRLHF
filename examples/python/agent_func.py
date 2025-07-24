import random
from typing import Any, Dict

import torch
from openrlhf.utils.agent import AgentExecutorBase, AgentInstanceBase


# A simple n-step random environment
class AgentInstance(AgentInstanceBase):
    def __init__(self, tokenizer):
        self.step_idx = 0
        self.max_steps = random.randint(1, 3)  # 1-3 steps
        self.tokenizer = tokenizer

    async def reset(self, states: dict, **kwargs):
        """Initialize the environment and return initial observation

        Args:
            states: Dictionary containing prompt and label

        Returns:
            dict: Dictionary containing observation (text) and observation_tokens (token ids)
        """
        prompt = states["prompt"]
        label = states["label"]

        # Tokenize the initial prompt using stored tokenizer
        observation_tokens = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"][
            0
        ].tolist()

        return {
            "observation": prompt,  # Original text observation
            "observation_tokens": observation_tokens,  # Tokenized observation
        }

    async def step(self, states: dict, **kwargs) -> Dict[str, Any]:
        """Execute one step of verification and return environment feedback

        Args:
            states: Dictionary containing observation_tokens, action_tokens, observation_text, action_text, and label

        Returns:
            Dict[str, Any]: A dictionary containing:
                - rewards: Reward value for advantage calculation
                - scores: Reward value for dynamic filtering
                - next_observation_tokens: The updated observation tokens after the step
                - done: Boolean indicating if the episode is complete
                - sampling_params: Parameters for vLLM sampling
                - extra_logs: Additional logging information
        """
        print(f"step_idx: {self.step_idx}, max_steps: {self.max_steps}")

        observation_tokens = states["observation_tokens"]
        action_tokens = states["action_tokens"]
        observation_text = states["observation_text"]
        action_text = states["action_text"]
        label = states["label"]

        # Check if episode is done
        done = self.step_idx >= self.max_steps
        reward = torch.randint(0, 2, (1,)).float() if done else torch.tensor(0)

        # Add feedback tokens based on whether episode is done
        feedback_text = (
            "\n\nHuman: [CORRECT]\n</s>"
            if done
            else "\n\nHuman: [INCORRECT]\nPlease analyze the issues and try again.\n</s>\n\nAssistant: "
        )
        feedback_tokens = self.tokenizer(feedback_text, add_special_tokens=False, return_tensors="pt")["input_ids"][
            0
        ].tolist()
        next_observation_tokens = observation_tokens + action_tokens + feedback_tokens

        self.step_idx += 1

        return {
            "rewards": reward,  # Rewards for advantage calculation
            "scores": reward,  # Scores for dynamic filtering (0-1 reward)
            "next_observation_tokens": next_observation_tokens,  # The updated observation tokens
            "done": done,  # Boolean indicating if the episode is complete
            "sampling_params": kwargs.get("sampling_params", None),  # Parameters for vLLM sampling in next step
            "extra_logs": {"dummy_scores": reward},  # Additional logging information
        }


class AgentExecutor(AgentExecutorBase):
    def __init__(self, max_steps, max_length, llm_engine, result_queue):
        super().__init__(AgentInstance, max_steps, max_length, llm_engine, result_queue)
