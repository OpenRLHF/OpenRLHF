import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_dynamic_filtering_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "openrlhf" / "trainer" / "ppo_utils" / "dynamic_filtering.py"
    spec = importlib.util.spec_from_file_location("openrlhf_dynamic_filtering", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_df = _load_dynamic_filtering_module()
should_keep_group = _df.should_keep_group
extract_group_scores = _df.extract_group_scores


class _MockScore:
    """Stand-in for a per-sample score tensor exposing ``.item()``."""

    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


def _mock_experiences(values):
    # ``scores`` mirrors the real Experience: indexable, element has ``.item()``.
    return [SimpleNamespace(scores=[_MockScore(v)] if v is not None else None) for v in values]


RANGE = (0.0, 1.0)


# ── should_keep_group ────────────────────────────────────────────────────────


def test_constant_continuous_reward_is_filtered():
    # Mean 0.5 is inside range, but zero variance -> no GRPO gradient.
    assert should_keep_group([0.5, 0.5, 0.5, 0.5], RANGE) is False


def test_varying_continuous_reward_is_kept():
    assert should_keep_group([0.2, 0.5, 0.7, 0.9], RANGE) is True


def test_binary_all_correct_is_filtered_by_range():
    # Mean == max_r, excluded by the strict range check (existing semantics).
    assert should_keep_group([1.0, 1.0, 1.0], RANGE) is False


def test_binary_all_wrong_is_filtered_by_range():
    assert should_keep_group([0.0, 0.0, 0.0], RANGE) is False


def test_binary_mixed_is_kept():
    assert should_keep_group([0.0, 1.0, 1.0, 0.0], RANGE) is True


def test_std_threshold_boundary():
    # Two symmetric points about mean 0.5 -> std == delta.
    scores = [0.5 - 1e-3, 0.5 + 1e-3]
    assert should_keep_group(scores, RANGE, std_threshold=1e-4) is True  # std above tol -> kept
    assert should_keep_group(scores, RANGE, std_threshold=1e-2) is False  # std below tol -> filtered


def test_default_threshold_keeps_tiny_variance():
    # Default 0.0 only filters exactly-constant groups.
    assert should_keep_group([0.5, 0.5 + 1e-9], RANGE) is True


# ── extract_group_scores ─────────────────────────────────────────────────────


def test_extract_scores_returns_values():
    assert extract_group_scores(_mock_experiences([0.1, 0.9])) == [0.1, 0.9]


def test_extract_scores_returns_none_when_missing():
    assert extract_group_scores(_mock_experiences([0.1, None])) is None


def test_extract_scores_returns_none_when_empty():
    assert extract_group_scores([]) is None
