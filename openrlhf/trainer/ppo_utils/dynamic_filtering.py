from typing import List, Optional, Tuple


def extract_group_scores(experiences) -> Optional[List[float]]:
    """Return the per-sample dynamic-filtering scores for a GRPO group.

    Returns ``None`` when the group is empty or any experience is missing a
    score, signalling that dynamic filtering should be skipped for this group.
    """
    if not experiences or any(e.scores is None for e in experiences):
        return None
    return [e.scores[0].item() for e in experiences]


def should_keep_group(
    scores: List[float],
    reward_range: Tuple[float, float],
    std_threshold: float = 0.0,
) -> bool:
    """Decide whether a GRPO group is worth keeping under dynamic filtering.

    A group is discarded when either:

    1. Its mean score falls outside the open interval ``reward_range`` (the
       existing DAPO range behavior; this also removes all-correct / all-wrong
       *binary* groups, whose mean is exactly ``max_r`` / ``min_r``), or
    2. Its score standard deviation is at or below ``std_threshold``.

    Rationale for (2): GRPO normalizes rewards within a group, so the advantage
    is ``A_i = (r_i - mean) / (std + eps)``. If every sample shares the same
    reward, ``A_i = 0`` for all ``i`` -> zero gradient, no learning signal, but
    the group still consumes generation and training compute. Binary rewards
    are already handled by the range check; the variance check closes the gap
    for *continuous* rewards (e.g. token-level F1), where a group can sit inside
    the range yet have zero variance.
    """
    min_r, max_r = reward_range
    mean = sum(scores) / len(scores)
    if not (min_r < mean < max_r):
        return False

    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    std = variance**0.5
    if std <= std_threshold:
        return False

    return True
