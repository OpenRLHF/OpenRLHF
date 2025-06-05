import random
from typing import Any, Dict

import torch

# Global states for the environment
step_idx = 0
max_steps = random.randint(0, 2)


# A n-step random environment
async def step(observation, action, label, **kwargs) -> Dict[str, Any]:
    """Execute one step of verification and return a random reward using torch.rand

    Args:
        observation: The input prompt/expression
        action: The language model's response
        label: Agent identifier or additional information

    Returns:
        Dict[str, Any]: A dictionary containing:
            - rewards: Reward value for advantage calculation
            - scores: Reward value for dynamic filtering
            - next_observation: The updated observation after the step
            - done: Boolean indicating if the episode is complete
            - sampling_params: Parameters for vLLM sampling
            - extra_logs: Additional logging information
    """
    global step_idx, max_steps
    print(f"step_idx: {step_idx}, max_steps: {max_steps}")

    # End after verification
    if step_idx >= max_steps:
        done = True
        # Generate a random reward using torch.rand
        reward = torch.randint(0, 2, (1,)).float()
        next_observation = (
            observation
            + action
            + "\n\nHuman: [VERIFICATION RESULT: CORRECT]\nYour solution is valid and complete. The verification process is finished.\n</s>"
        )
    else:
        done = False
        reward = torch.tensor(0)
        # Update observation
        next_observation = (
            observation
            + action
            + "\n\nHuman: [VERIFICATION RESULT: INCORRECT]\nLet's analyze what needs improvement:\n1. What are the key issues in the current solution?\n2. How can we make it more robust?\n3. What additional considerations should we take into account?\n\nPlease provide your revised solution:\n</s>\n\nAssistant: "
        )
    step_idx += 1

    return {
        "rewards": reward,  # Rewards for advantage calculation
        "scores": reward,  # Scores for dynamic filtering (0-1 reward)
        "next_observation": next_observation,  # The updated observation for vLLM inference in next step
        "done": done,  # Boolean indicating if the episode is complete
        "sampling_params": kwargs.get("sampling_params", None),  # Parameters for vLLM sampling in next step
        "extra_logs": {"dummy_scores": reward},  # Additional logging information
    }
