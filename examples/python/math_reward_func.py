import re
from typing import List, Optional, Tuple

import ray
import torch

# math-verify is required: pip install 'math-verify[antlr4_13_2]'
from math_verify import parse, verify

_VERIFY_TIMEOUT = 180  # seconds per verification


@ray.remote
def _verify_with_ray(pred_answer: str, gold_answer: str) -> Tuple[bool, Optional[str]]:
    """
    Ray remote function for math verification.
    Runs in a separate Ray worker process, allowing proper timeout handling.

    Returns:
        Tuple of (is_correct, error_message)
    """
    try:
        parsed_gold = parse(gold_answer)
        parsed_pred = parse(pred_answer)
        return verify(parsed_gold, parsed_pred), None
    except Exception as e:
        return False, str(e)


def extract_boxed_answer(text: str) -> Optional[str]:
    """
    Extract the answer from \\boxed{} in the text.
    If multiple \\boxed{} are found, return the last one.

    Args:
        text: The text containing \\boxed{answer}

    Returns:
        The extracted answer or None if not found
    """
    if text is None:
        return None

    # Match \boxed{...} with nested braces support
    pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()

    # Fallback: simple pattern without nested braces
    simple_pattern = r"\\boxed\{([^}]+)\}"
    simple_matches = re.findall(simple_pattern, text)
    if simple_matches:
        return simple_matches[-1].strip()

    return None


def verify_math_answer(pred_answer: str, gold_answer: str) -> bool:
    """
    Verify if the predicted answer matches the gold answer using math-verify library.
    Runs verification in a Ray worker process with timeout support.

    Args:
        pred_answer: The predicted answer string
        gold_answer: The gold/reference answer string

    Returns:
        True if answers match, False otherwise
    """
    if pred_answer is None or gold_answer is None:
        return False

    try:
        ref = _verify_with_ray.remote(pred_answer, gold_answer)
        is_correct, error = ray.get(ref, timeout=_VERIFY_TIMEOUT)

        if error:
            print(f"[Math Verify] Parse error: {error}, pred={pred_answer}, gold={gold_answer}")
        return is_correct
    except ray.exceptions.GetTimeoutError:
        print(f"[Math Verify] Timeout, pred={pred_answer}, gold={gold_answer}")
        return False
    except Exception as e:
        print(f"[Math Verify] Error: {e}, pred={pred_answer}, gold={gold_answer}")
        return False


def reward_func(queries: List[str], prompts: List[str], labels: List[str], **kwargs) -> dict:
    """
    Reward function for verifying math answers using math-verify library.

    This function extracts answers from \\boxed{} in the model's responses
    and compares them with the ground truth labels using the math-verify library.

    Args:
        queries: Complete text sequences containing prompts and responses
        prompts: Input prompt sequences
        labels: Ground truth answer sequences
        **kwargs: Additional optional parameters

    Returns:
        dict: A dictionary containing:
            - rewards: Reward values (1.0 for correct, 0.0 for incorrect)
            - scores: Same as rewards, used for dynamic filtering
            - extra_logs: Additional logging information
    """
    rewards = []

    for query, prompt, label in zip(queries, prompts, labels):
        # Extract the response part (after the prompt)
        if isinstance(prompt, str) and prompt in query:
            response = query[len(prompt) :]
        else:
            response = query

        # Extract answer from response
        pred_answer = extract_boxed_answer(response)
        gold_answer = label

        # Verify if answers match using math-verify
        is_correct = verify_math_answer(pred_answer, gold_answer)

        rewards.append(1.0 if is_correct else 0.0)
        print(f"[Math Reward] Pred: {pred_answer}, Gold: {gold_answer}, Match: {is_correct}")

    rewards_tensor = torch.tensor(rewards, dtype=torch.float)
    accuracy = rewards_tensor.mean()

    return {
        "rewards": rewards_tensor,
        "scores": rewards_tensor,
        "extra_logs": {
            "math_accuracy": accuracy,
        },
    }
