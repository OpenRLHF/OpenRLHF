import random
from typing import Any, Dict, Tuple

import torch

step_idx = 0
max_steps = random.randint(0, 2)


# A n-step random environment
async def step(state, action, label, **kwargs) -> Tuple[float, Dict[str, Any], bool]:
    """Executes one step of verification and returns a random reward using torch.rand

    Args:
        state: The prompt/input expression
        action: The response from the language model
        label: Used to identify the agent / pass extra info

    Returns:
        Tuple[float, Dict[str, Any], bool]: (random_reward, next_state, done)
    """
    global step_idx, max_steps
    print(f"step_idx: {step_idx}, max_steps: {max_steps}")

    # Update state
    next_state = state + action

    # End after verification
    if step_idx >= max_steps:
        done = True
        # Generate a random reward using torch.rand
        reward = torch.rand(1)
    else:
        done = False
        reward = torch.tensor(0)
    step_idx += 1

    # Extra info
    extra_info = {}

    return reward, next_state, done, extra_info
