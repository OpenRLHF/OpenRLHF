import importlib.util
import sys
from pathlib import Path

import pytest
import torch


def _load_constrained_reward_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "openrlhf" / "utils" / "constrained_reward.py"
    spec = importlib.util.spec_from_file_location("_constrained_reward_test_module", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_constrained_reward_test_module"] = module
    spec.loader.exec_module(module)
    return module


_constrained_reward = _load_constrained_reward_module()

AllTrueJudge = _constrained_reward.AllTrueJudge
BaseBinaryJudge = _constrained_reward.BaseBinaryJudge
calibrated_reward = _constrained_reward.calibrated_reward
shape_constrained_rewards = _constrained_reward.shape_constrained_rewards


class StaticJudge(BaseBinaryJudge):
    def __init__(self, judgments):
        self.judgments = judgments

    def judge(self, prompts, completions, gold_completions=None):
        return self.judgments


class ContainsGoldJudge(BaseBinaryJudge):
    def judge(self, prompts, completions, gold_completions=None):
        if gold_completions is None:
            raise ValueError("gold_completions is required")

        return [
            int(str(gold).strip().lower() in str(completion).strip().lower())
            for completion, gold in zip(completions, gold_completions)
        ]


def test_calibrated_reward_matches_sigmoid_difference():
    rewards = torch.tensor([3.0, 1.0])
    baseline_rewards = torch.tensor([1.0, 2.0])

    result = calibrated_reward(rewards, baseline_rewards)

    expected = torch.sigmoid(torch.tensor([2.0, -1.0]))
    assert torch.allclose(result, expected)


def test_calibrated_reward_supports_temperature():
    rewards = torch.tensor([3.0])
    baseline_rewards = torch.tensor([1.0])

    result = calibrated_reward(rewards, baseline_rewards, temperature=2.0)

    expected = torch.sigmoid(torch.tensor([1.0]))
    assert torch.allclose(result, expected)


def test_calibrated_reward_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        calibrated_reward([1.0, 2.0], [1.0])


def test_calibrated_reward_rejects_non_positive_temperature():
    with pytest.raises(ValueError, match="temperature"):
        calibrated_reward([1.0], [0.0], temperature=0.0)


def test_all_true_judge_combines_binary_constraints():
    judge = AllTrueJudge(
        [
            StaticJudge([1, 1, 1, -1]),
            StaticJudge([1, 0, -1, 1]),
        ]
    )

    result = judge.judge(
        prompts=["p0", "p1", "p2", "p3"],
        completions=["c0", "c1", "c2", "c3"],
    )

    assert result == [1, 0, -1, -1]


def test_all_true_judge_requires_at_least_one_judge():
    with pytest.raises(ValueError, match="at least one judge"):
        AllTrueJudge([])


def test_all_true_judge_rejects_invalid_judgment():
    judge = AllTrueJudge([StaticJudge([2])])

    with pytest.raises(ValueError, match="Binary judgments"):
        judge.judge(prompts=["p"], completions=["c"])


def test_shape_constrained_rewards_applies_constraint_penalty():
    result = shape_constrained_rewards(
        prompts=["p0", "p1", "p2"],
        completions=["c0", "c1", "c2"],
        rewards=[1.0, 1.0, 1.0],
        judge=StaticJudge([1, 0, -1]),
        constraint_penalty=0.25,
        judge_error_penalty=0.5,
    )

    assert result.rewards == pytest.approx([1.0, 0.75, 0.5])
    assert result.scores == [1, 0, 0]
    assert result.extra_logs["constraint_judgment"] == [1, 0, -1]
    assert result.extra_logs["constraint_penalty_applied"] == [0.0, 0.25, 0.5]


def test_shape_constrained_rewards_applies_calibration_and_penalty():
    result = shape_constrained_rewards(
        prompts=["p0", "p1"],
        completions=["good", "bad"],
        rewards=[3.0, 3.0],
        gold_completions=["good", "good"],
        baseline_rewards=[1.0, 1.0],
        judge=StaticJudge([1, 0]),
        constraint_penalty=0.5,
        calibrate=True,
    )

    expected = torch.sigmoid(torch.tensor([2.0, 2.0]))

    assert result.scores == [1, 0]
    assert result.rewards[0] == pytest.approx(float(expected[0]))
    assert result.rewards[1] == pytest.approx(float(expected[1] - 0.5))
    assert result.extra_logs["constraint_judgment"] == [1, 0]


def test_shape_constrained_rewards_requires_baseline_for_calibration():
    with pytest.raises(ValueError, match="baseline_rewards"):
        shape_constrained_rewards(
            prompts=["p"],
            completions=["c"],
            rewards=[1.0],
            calibrate=True,
        )


def test_shape_constrained_rewards_can_use_gold_completion_judge():
    result = shape_constrained_rewards(
        prompts=["What is 2+2?", "What is 3+3?"],
        completions=["The answer is 4.", "The answer is 7."],
        rewards=[1.0, 1.0],
        gold_completions=["4", "6"],
        judge=ContainsGoldJudge(),
        constraint_penalty=1.0,
    )

    assert result.rewards == pytest.approx([1.0, 0.0])
    assert result.scores == [1, 0]
