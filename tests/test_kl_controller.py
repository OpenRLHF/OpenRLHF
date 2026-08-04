import importlib.util
from pathlib import Path

import pytest


def _load_kl_controller_module():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "openrlhf.trainer.ppo_utils.kl_controller",
        root / "openrlhf" / "trainer" / "ppo_utils" / "kl_controller.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AdaptiveKLController = _load_kl_controller_module().AdaptiveKLController


@pytest.mark.parametrize(
    ("target", "horizon", "message"),
    [(0.0, 100, "target"), (-1.0, 100, "target"), (1.0, 0, "horizon"), (1.0, -1, "horizon")],
)
def test_adaptive_kl_controller_rejects_non_positive_parameters(target, horizon, message):
    with pytest.raises(ValueError, match=message):
        AdaptiveKLController(init_kl_coef=0.1, target=target, horizon=horizon)


def test_adaptive_kl_controller_updates_with_valid_parameters():
    controller = AdaptiveKLController(init_kl_coef=0.1, target=1.0, horizon=100)

    controller.update(current=1.2, n_steps=10)

    assert controller.value == pytest.approx(0.102)
