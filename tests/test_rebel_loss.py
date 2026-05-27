"""Unit tests for REBELLoss.

Directly load loss.py to avoid importing openrlhf.models and optional GPU deps.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

# --- Direct-load loss.py without importing the openrlhf.models package ---------
_REPO_ROOT = Path(__file__).resolve().parents[1]
_MODELS_DIR = _REPO_ROOT / "openrlhf" / "models"
_TEST_PACKAGE = "_openrlhf_rebel_loss_test"


def _load_loss_module():
    pkg = types.ModuleType(_TEST_PACKAGE)
    pkg.__path__ = [str(_MODELS_DIR)]
    sys.modules[_TEST_PACKAGE] = pkg
    for name in ("utils", "loss"):
        spec = importlib.util.spec_from_file_location(
            f"{_TEST_PACKAGE}.{name}", _MODELS_DIR / f"{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"{_TEST_PACKAGE}.{name}"] = module
        spec.loader.exec_module(module)
    return sys.modules[f"{_TEST_PACKAGE}.loss"]


_loss_mod = _load_loss_module()
REBELLoss = _loss_mod.REBELLoss


# --- Helpers -------------------------------------------------------------------
def _make_logps(pi_chosen, pi_rejected, old_chosen, old_rejected):
    """Build four logp tensors from python lists."""
    return (
        torch.tensor(pi_chosen, dtype=torch.float32),
        torch.tensor(pi_rejected, dtype=torch.float32),
        torch.tensor(old_chosen, dtype=torch.float32),
        torch.tensor(old_rejected, dtype=torch.float32),
    )


def _logits(pi_c, pi_r, old_c, old_r):
    """The double-difference the loss regresses on: (pi_c - pi_r) - (old_c - old_r)."""
    return (pi_c - pi_r) - (old_c - old_r)


# --- Tests ---------------------------------------------------------------------
def test_invalid_eta_raises():
    """eta must be > 0; zero and negative values are rejected."""
    with pytest.raises(ValueError, match="eta must be > 0"):
        REBELLoss(0.0)
    with pytest.raises(ValueError, match="eta must be > 0"):
        REBELLoss(-1.0)
    
def test_zero_loss_at_target():
    """When logits == eta * reward_margin exactly, the loss is ~0.

    Construct logps so logits = 2.0 for every element, and pick margin so that
    eta * margin = 2.0. The squared residual should then vanish.
    """
    eta = 4.0
    loss_fn = REBELLoss(eta)
    pi_c, pi_r, old_c, old_r = _make_logps([3.0, 3.0], [0.0, 0.0], [1.0, 1.0], [0.0, 0.0])
    logits = _logits(pi_c, pi_r, old_c, old_r)
    reward_margin = logits / eta 

    loss, _, _ = loss_fn(pi_c, pi_r, old_c, old_r, reward_margin)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6), loss
    
def test_positive_loss_off_target():
    """When logits != eta * reward_margin, loss equals the squared residual."""
    eta = 1.0
    loss_fn = REBELLoss(eta)
    pi_c, pi_r, old_c, old_r = _make_logps([2.0], [0.0], [0.0], [0.0])
    reward_margin = torch.tensor([1.0], dtype=torch.float32)
    loss, _, _ = loss_fn(pi_c, pi_r, old_c, old_r, reward_margin)
    assert torch.isclose(loss, torch.tensor(1.0), atol=1e-6), loss
    
def test_returned_rewards_difference_equals_logits():
    """(chosen_reward - rejected_reward) == logits exactly."""
    eta = 7.5
    loss_fn = REBELLoss(eta)
    pi_c, pi_r, old_c, old_r = _make_logps([1.5, 4.0], [0.5, 1.0], [0.2, 0.5], [0.1, 0.3])
    reward_margin = torch.tensor([0.3, 0.6], dtype=torch.float32)

    _, chosen_reward, rejected_reward = loss_fn(pi_c, pi_r, old_c, old_r, reward_margin)
    pred_gap = chosen_reward - rejected_reward
    logits = _logits(pi_c, pi_r, old_c, old_r)
    assert torch.allclose(pred_gap, logits, atol=1e-6), (pred_gap, logits)

def test_acc_reduces_to_sign_of_logits():
    """chosen_reward > rejected_reward iff logits > 0, independent of eta."""
    for eta in (0.5, 3.0, 15.0):
        loss_fn = REBELLoss(eta)
        pi_c, pi_r, old_c, old_r = _make_logps([2.0, 0.0], [0.0, 2.0], [0.0, 0.0], [0.0, 0.0])
        reward_margin = torch.tensor([0.4, 0.4], dtype=torch.float32)
        _, chosen_reward, rejected_reward = loss_fn(pi_c, pi_r, old_c, old_r, reward_margin)
        logits = _logits(pi_c, pi_r, old_c, old_r)
        assert torch.equal(chosen_reward > rejected_reward, logits > 0)

def test_gradient_step_moves_logits_toward_target():
    """One SGD step moves logits toward eta * reward_margin."""
    eta = 0.5
    loss_fn = REBELLoss(eta)
    pi_c = torch.tensor([0.0], dtype=torch.float32, requires_grad=True)
    pi_r = torch.tensor([0.0], dtype=torch.float32, requires_grad=True)
    old_c = torch.tensor([0.0], dtype=torch.float32)
    old_r = torch.tensor([0.0], dtype=torch.float32)
    reward_margin = torch.tensor([2.0], dtype=torch.float32)

    logits_before = (pi_c - pi_r).item()
    optimizer = torch.optim.SGD([pi_c, pi_r], lr=0.1)
    loss, _, _ = loss_fn(pi_c, pi_r, old_c, old_r, reward_margin)
    loss.backward()
    optimizer.step()
    logits_after = (pi_c - pi_r).item()

    target = eta * 2.0
    assert logits_after > logits_before
    assert logits_after <= target