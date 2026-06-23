from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence

import torch

_VALID_BINARY_JUDGMENTS = {-1, 0, 1}


class BaseBinaryJudge(ABC):
    """Base class for binary constraint judges.

    A judge returns:
    - 1 when the completion satisfies the constraint,
    - 0 when the completion violates the constraint,
    - -1 when judging failed.
    """

    @abstractmethod
    def judge(
        self,
        prompts: Sequence[str],
        completions: Sequence[str],
        gold_completions: Optional[Sequence[str]] = None,
    ) -> list[int]:
        raise NotImplementedError


class AllTrueJudge(BaseBinaryJudge):
    """Combine binary judges using Mixture-of-Judges / all-constraints logic.

    Returns:
    - 1 if every judge returns 1,
    - 0 if any judge returns 0,
    - -1 if at least one judge returns -1 and no judge returns 0.
    """

    def __init__(self, judges: Sequence[BaseBinaryJudge]) -> None:
        if not judges:
            raise ValueError("AllTrueJudge requires at least one judge.")
        self.judges = list(judges)

    def judge(
        self,
        prompts: Sequence[str],
        completions: Sequence[str],
        gold_completions: Optional[Sequence[str]] = None,
    ) -> list[int]:
        _check_same_length("prompts", prompts, "completions", completions)
        if gold_completions is not None:
            _check_same_length("prompts", prompts, "gold_completions", gold_completions)

        per_judge_results = []
        for judge in self.judges:
            judgments = judge.judge(prompts, completions, gold_completions)
            _validate_binary_judgments(judgments)
            _check_same_length("prompts", prompts, "judgments", judgments)
            per_judge_results.append(judgments)

        combined = []
        for sample_judgments in zip(*per_judge_results):
            if any(judgment == 0 for judgment in sample_judgments):
                combined.append(0)
            elif any(judgment == -1 for judgment in sample_judgments):
                combined.append(-1)
            else:
                combined.append(1)

        return combined


@dataclass
class ConstrainedRewardResult:
    rewards: list[float]
    scores: Optional[list[int]]
    extra_logs: dict[str, list[float | int]]


def calibrated_reward(
    rewards: Sequence[float] | torch.Tensor,
    baseline_rewards: Sequence[float] | torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute calibrated rewards against a baseline/reference completion.

    R_calib = sigmoid((R_generated - R_baseline) / temperature)
    """

    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    rewards_tensor = _as_float_tensor(rewards)
    baseline_tensor = _as_float_tensor(baseline_rewards).to(rewards_tensor.device)

    if rewards_tensor.shape != baseline_tensor.shape:
        raise ValueError(
            "rewards and baseline_rewards must have the same shape, got "
            f"{tuple(rewards_tensor.shape)} and {tuple(baseline_tensor.shape)}"
        )

    return torch.sigmoid((rewards_tensor - baseline_tensor) / temperature)


def shape_constrained_rewards(
    *,
    prompts: Sequence[str],
    completions: Sequence[str],
    rewards: Sequence[float] | torch.Tensor,
    gold_completions: Optional[Sequence[str]] = None,
    baseline_rewards: Optional[Sequence[float] | torch.Tensor] = None,
    judge: Optional[BaseBinaryJudge] = None,
    constraint_penalty: float = 1.0,
    judge_error_penalty: Optional[float] = None,
    calibrate: bool = False,
    temperature: float = 1.0,
) -> ConstrainedRewardResult:
    """Apply optional calibrated reward and binary constraint penalties.

    This is designed for custom OpenRLHF reward functions loaded through
    --reward.remote_url path/to/reward_file.py.
    """

    _check_same_length("prompts", prompts, "completions", completions)
    _check_same_length("prompts", prompts, "rewards", rewards)

    if gold_completions is not None:
        _check_same_length("prompts", prompts, "gold_completions", gold_completions)

    if calibrate:
        if baseline_rewards is None:
            raise ValueError("baseline_rewards is required when calibrate=True")
        shaped = calibrated_reward(rewards, baseline_rewards, temperature=temperature)
    else:
        shaped = _as_float_tensor(rewards)

    extra_logs: dict[str, list[float | int]] = {
        "raw_reward": _as_float_tensor(rewards).detach().cpu().tolist(),
    }

    if calibrate:
        extra_logs["baseline_reward"] = _as_float_tensor(baseline_rewards).detach().cpu().tolist()
        extra_logs["calibrated_reward"] = shaped.detach().cpu().tolist()

    scores = None
    if judge is not None:
        judgments = judge.judge(prompts, completions, gold_completions)
        _validate_binary_judgments(judgments)
        _check_same_length("prompts", prompts, "judgments", judgments)

        if judge_error_penalty is None:
            judge_error_penalty = constraint_penalty

        judgment_tensor = torch.tensor(judgments, dtype=torch.long, device=shaped.device)
        violation_mask = judgment_tensor == 0
        error_mask = judgment_tensor == -1

        shaped = shaped.clone()
        shaped[violation_mask] -= constraint_penalty
        shaped[error_mask] -= judge_error_penalty

        scores = [int(judgment == 1) for judgment in judgments]
        extra_logs["constraint_judgment"] = judgments
        extra_logs["constraint_penalty_applied"] = (
            violation_mask.float() * constraint_penalty + error_mask.float() * judge_error_penalty
        ).tolist()

    return ConstrainedRewardResult(
        rewards=[float(value) for value in shaped.detach().cpu().tolist()],
        scores=scores,
        extra_logs=extra_logs,
    )

def _as_float_tensor(values: Sequence[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values.detach().float()
    return torch.tensor(list(values), dtype=torch.float32)


def _check_same_length(left_name: str, left, right_name: str, right) -> None:
    if len(left) != len(right):
        raise ValueError(f"{left_name} and {right_name} must have the same length, got {len(left)} and {len(right)}")


def _validate_binary_judgments(judgments: Sequence[int]) -> None:
    invalid = [judgment for judgment in judgments if judgment not in _VALID_BINARY_JUDGMENTS]
    if invalid:
        raise ValueError(f"Binary judgments must be in {{-1, 0, 1}}, got invalid values: {invalid}")
