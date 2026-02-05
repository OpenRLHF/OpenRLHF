"""Math reward function for verifying mathematical answers.

Supports two answer formats:
1. \\boxed{answer} - LaTeX boxed format
2. Answer: answer - Plain text format (e.g., "Answer: 42")
"""

import re
from typing import List, Optional

import torch

from openrlhf.utils import extract_boxed_answer, grade_answer


def extract_answer_tag(response: str) -> Optional[str]:
    """Extract answer from 'Answer: xxx' format.

    Looks for the last occurrence of 'Answer:' followed by the answer on the same line.
    """
    # Find all matches of "Answer:" followed by content
    pattern = r"Answer:\s*(.+?)(?:\n|$)"
    matches = re.findall(pattern, response, re.IGNORECASE)
    if matches:
        # Return the last match, stripped of whitespace
        return matches[-1].strip()
    return None


def extract_model_answer(response: str) -> Optional[str]:
    """Extract answer from model response.

    Priority:
    1. Try 'Answer: xxx' format first (for DAPO-style prompts)
    2. Fall back to \\boxed{xxx} format
    """
    # First try Answer: format
    answer = extract_answer_tag(response)
    if answer is not None:
        return answer

    # Fall back to boxed format
    return extract_boxed_answer(response)


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

        # Extract and grade the answer
        pred_answer = extract_model_answer(response)
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
