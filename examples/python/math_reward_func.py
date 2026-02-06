"""Math reward function for verifying mathematical answers.

Only supports \\boxed{answer} LaTeX format for answer extraction.
"""

from typing import List

import torch

from openrlhf.utils import extract_boxed_answer, grade_answer


def reward_func(queries: List[str], prompts: List[str], labels: List[str], **kwargs) -> dict:
    """
    Reward function for verifying math answers.

    Args:
        queries: Complete text sequences containing prompts and responses
        prompts: Input prompt sequences
        labels: Ground truth answer sequences
        **kwargs: Additional optional parameters

    Returns:
        dict with rewards, scores, and extra_logs
    """
    rewards = []

    for query, prompt, label in zip(queries, prompts, labels):
        # Extract the response part (after the prompt)
        if isinstance(prompt, str) and prompt in query:
            response = query[len(prompt) :]
        else:
            response = query

        # Extract and grade the answer (only boxed format)
        pred_answer = extract_boxed_answer(response)
        is_correct = grade_answer(pred_answer, label)
        rewards.append(1.0 if is_correct else 0.0)

        # Debug logging
        print(f"[Math Reward] Pred: {pred_answer}, Gold: {label}, Match: {is_correct}")

    rewards_tensor = torch.tensor(rewards, dtype=torch.float)
    accuracy = rewards_tensor.mean()

    return {
        "rewards": rewards_tensor,
        "scores": rewards_tensor,
        "extra_logs": {
            "math_accuracy": accuracy,
        },
    }
