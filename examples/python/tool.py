from typing import Any, Dict, Tuple

import torch


def step(state, action, **kwargs) -> Tuple[float, Dict[str, Any], bool]:
    """Executes one step of verification and returns a random reward using torch.rand

    Args:
        state: The prompt/input expression
        action: Mathematical expression to verify

    Returns:
        Tuple[float, Dict[str, Any], bool]: (random_reward, next_state, done)
    """
    print(f"state: {state}, action: {action}")

    # Generate a random reward using torch.rand
    reward = torch.rand(1)

    # Update state
    next_state = None

    # End after verification
    done = True

    # Extra info
    extra_info = {}

    return reward, next_state, done, extra_info
