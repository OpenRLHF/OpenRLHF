import importlib.util
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def lp(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "openrlhf.utils.logging_utils",
        SimpleNamespace(init_logger=logging.getLogger),
    )
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "length_penalty", root / "openrlhf" / "trainer" / "ppo_utils" / "length_penalty.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_exp(response_length, total_length, reward=0.0):
    return SimpleNamespace(
        response_length=np.array([response_length]),
        total_length=np.array([total_length]),
        rewards=np.array([reward]),
        info={},
    )


class TestOverlongPerSampleLimit:
    """When max_new_tokens is None the per-sample generation budget is
    max_len - prompt_len.  The penalty must use that budget, not max_len."""

    def test_full_budget_response_gets_max_penalty(self, lp):
        """A response that fills its per-sample budget should receive the full
        penalty (-1.0 * factor), regardless of prompt length."""
        max_len, buf, factor = 2048, 512, 1.0
        for prompt_len in (100, 500, 1000, 1500):
            budget = max_len - prompt_len
            exp = _make_exp(
                response_length=budget,
                total_length=max_len,
            )
            lp.apply_overlong_penalty(
                [exp],
                max_new_tokens=None,
                overlong_buffer_len=buf,
                overlong_penalty_factor=factor,
                max_len=max_len,
            )
            assert exp.rewards[0] == pytest.approx(-1.0), (
                f"prompt_len={prompt_len}: expected -1.0, got {exp.rewards[0]}"
            )

    def test_short_response_no_penalty(self, lp):
        """A response well below the threshold should not be penalized."""
        exp = _make_exp(response_length=100, total_length=600)
        lp.apply_overlong_penalty(
            [exp],
            max_new_tokens=None,
            overlong_buffer_len=512,
            overlong_penalty_factor=1.0,
            max_len=2048,
        )
        assert exp.rewards[0] == 0.0

    def test_budget_smaller_than_buffer_skips_penalty(self, lp):
        """When the prompt is so long that the per-sample budget is smaller
        than overlong_buffer_len, no penalty should be applied."""
        exp = _make_exp(response_length=400, total_length=2048)
        lp.apply_overlong_penalty(
            [exp],
            max_new_tokens=None,
            overlong_buffer_len=512,
            overlong_penalty_factor=1.0,
            max_len=2048,
        )
        assert exp.rewards[0] == 0.0

    def test_explicit_max_new_tokens_unchanged(self, lp):
        """When max_new_tokens is set explicitly, behavior is unchanged."""
        exp = _make_exp(response_length=950, total_length=1950)
        lp.apply_overlong_penalty(
            [exp],
            max_new_tokens=1024,
            overlong_buffer_len=512,
            overlong_penalty_factor=1.0,
        )
        expected = -(min(950 - (1024 - 512), 512)) / 512
        assert exp.rewards[0] == pytest.approx(expected)

    def test_apply_length_penalties_per_sample(self, lp):
        """Integration: apply_length_penalties with max_new_tokens=None."""
        args = SimpleNamespace(
            rollout=SimpleNamespace(max_new_tokens=None),
            data=SimpleNamespace(max_len=2048),
            reward=SimpleNamespace(overlong_buffer_len=512, overlong_penalty_factor=1.0),
        )
        exps = [
            _make_exp(response_length=1948, total_length=2048),
            _make_exp(response_length=548, total_length=2048),
        ]
        lp.apply_length_penalties(exps, args)
        assert exps[0].rewards[0] == pytest.approx(-1.0)
        assert exps[1].rewards[0] == pytest.approx(-1.0)
