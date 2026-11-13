"""
Length Penalty Module for RLHF Training

Two types of length penalty are supported:
1. DAPO Overlong Penalty: Penalizes based on response length exceeding a threshold
2. ProRL Stop Properly Penalty: Penalizes truncated samples (finish_reason == "length")
"""

from typing import List

import torch

from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)


def apply_overlong_penalty(
    experiences: List,
    max_new_tokens: int,
    overlong_buffer_len: float,
    overlong_penalty_factor: float = 1.0,
) -> int:
    """
    DAPO-style overlong penalty based on response length.

    Penalizes responses that exceed (max_new_tokens - overlong_buffer_len).
    Formula: penalty = -min(exceed_len, buffer_len) / buffer_len * penalty_factor

    Args:
        experiences: List of Experience objects with rewards and info
        max_new_tokens: Maximum generation length
        overlong_buffer_len: Buffer length before max_new_tokens
        overlong_penalty_factor: Maximum penalty factor

    Returns:
        Number of samples that received penalty
    """
    assert (
        max_new_tokens >= overlong_buffer_len
    ), f"max_new_tokens ({max_new_tokens}) must be >= overlong_buffer_len ({overlong_buffer_len})"

    expected_len = max_new_tokens - overlong_buffer_len
    total_penalized = 0

    if overlong_buffer_len <= 0:
        return total_penalized

    for experience in experiences:
        # Match the Python scalar calculation before rounding the penalty to the reward dtype.
        exceed_len = (experience.response_length.detach().to(torch.float64) - expected_len).clamp(
            max=overlong_buffer_len
        )
        penalized = exceed_len > 0
        num_penalized = int(penalized.sum().item())
        if num_penalized:
            penalties = -exceed_len[penalized] / overlong_buffer_len * overlong_penalty_factor
            penalties = penalties.to(experience.rewards.device)
            if experience.rewards.is_floating_point():
                penalties = penalties.to(experience.rewards.dtype)
            experience.rewards[penalized.to(experience.rewards.device)] += penalties
            total_penalized += num_penalized

    return total_penalized


def apply_stop_properly_penalty(
    experiences: List,
    stop_properly_penalty_coef: float,
) -> int:
    """
    ProRL-style stop properly penalty based on vLLM finish_reason.

    Penalizes samples that were truncated (finish_reason == "length").

    When stop_properly_penalty_coef >= 0: scale truncated rewards by this coefficient.
    When stop_properly_penalty_coef < 0: set truncated rewards to this value directly
        (e.g., -0.5 gives truncated samples a fixed negative reward).

    Args:
        experiences: List of Experience objects with rewards and info
        stop_properly_penalty_coef: Coefficient to penalize truncated samples.
            If >= 0: multiplicative scaling [0, 1].
            If < 0: fixed reward override for truncated samples.

    Returns:
        Number of truncated samples
    """
    if stop_properly_penalty_coef >= 0:
        assert (
            0 <= stop_properly_penalty_coef <= 1
        ), f"stop_properly_penalty_coef must be in [0, 1] or negative, got {stop_properly_penalty_coef}"

    total_truncated = 0

    for experience in experiences:
        truncated_flags = experience.truncated
        if truncated_flags is None:
            continue

        truncated = truncated_flags.to(device=experience.rewards.device, dtype=torch.bool)
        num_truncated = int(truncated.sum().item())
        if num_truncated:
            if stop_properly_penalty_coef < 0:
                experience.rewards[truncated] = stop_properly_penalty_coef
            else:
                experience.rewards[truncated] = (experience.rewards[truncated] * stop_properly_penalty_coef).to(
                    experience.rewards.dtype
                )
            total_truncated += num_truncated

    return total_truncated


def apply_length_penalties(experiences: List, args) -> None:
    """
    Apply length penalties to experiences based on configuration.

    Supports two types of penalties:
    1. DAPO Overlong Penalty (--overlong_buffer_len, --overlong_penalty_factor)
    2. ProRL Stop Properly Penalty (--stop_properly_penalty_coef)

    Both can be enabled simultaneously.

    Args:
        experiences: List of Experience objects
        args: Training arguments containing penalty configuration
    """
    total_samples = sum(len(exp.rewards) for exp in experiences)

    # DAPO-style overlong penalty based on response length
    if getattr(args.reward, "overlong_buffer_len", None) is not None:
        max_new_tokens = getattr(args.rollout, "max_new_tokens", None) or args.data.max_len
        num_penalized = apply_overlong_penalty(
            experiences=experiences,
            max_new_tokens=max_new_tokens,
            overlong_buffer_len=args.reward.overlong_buffer_len,
            overlong_penalty_factor=getattr(args.reward, "overlong_penalty_factor", 1.0),
        )
        logger.info(
            f"[DAPO Overlong Penalty] {num_penalized}/{total_samples} samples penalized, "
            f"buffer_len={args.reward.overlong_buffer_len}, factor={args.reward.overlong_penalty_factor}"
        )

    # ProRL-style stop properly penalty based on finish_reason
    if getattr(args.reward, "stop_properly_penalty_coef", None) is not None:
        num_truncated = apply_stop_properly_penalty(
            experiences=experiences,
            stop_properly_penalty_coef=args.reward.stop_properly_penalty_coef,
        )
        logger.info(
            f"[ProRL Stop Properly Penalty] {num_truncated}/{total_samples} samples truncated, "
            f"coef={args.reward.stop_properly_penalty_coef}"
        )

    # Sync info["reward"] with the modified rewards so logged metrics reflect penalties
    for experience in experiences:
        if "reward" in experience.info:
            experience.info["reward"] = experience.rewards.clone()
