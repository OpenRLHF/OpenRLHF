import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

_TEST_PACKAGE = "_openrlhf_kl_test"


def _load_utils_module():
    root = Path(__file__).resolve().parents[1]
    models_dir = root / "openrlhf" / "models"

    pkg = types.ModuleType(_TEST_PACKAGE)
    pkg.__path__ = [str(models_dir)]
    sys.modules[_TEST_PACKAGE] = pkg

    spec = importlib.util.spec_from_file_location(f"{_TEST_PACKAGE}.utils", models_dir / "utils.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"{_TEST_PACKAGE}.utils"] = module
    spec.loader.exec_module(module)
    return module


_utils = _load_utils_module()
compute_approx_kl = _utils.compute_approx_kl
KL_IS_WEIGHT_CLIP = _utils.KL_IS_WEIGHT_CLIP

ESTIMATORS = ["k1", "k2", "k3"]


def _grad_wrt_log_probs(log_probs, log_probs_base, **kwargs):
    lp = log_probs.clone().detach().requires_grad_(True)
    kl = compute_approx_kl(lp, log_probs_base, **kwargs)
    kl.sum().backward()
    return lp.grad


def test_flag_off_matches_legacy():
    # With unbiased_gradient=False the output must equal the legacy clamped estimator, byte-for-byte.
    torch.manual_seed(42)
    log_probs = torch.randn(4, 8)
    log_probs_base = torch.randn(4, 8)
    for est in ESTIMATORS:
        log_ratio = log_probs.float() - log_probs_base.float()
        if est == "k1":
            legacy = log_ratio
        elif est == "k2":
            legacy = log_ratio**2 / 2.0
        else:
            legacy = (-log_ratio).exp() - 1 + log_ratio
        legacy = legacy.clamp(min=-10, max=10)
        got = compute_approx_kl(log_probs, log_probs_base, kl_estimator=est, unbiased_gradient=False)
        assert torch.allclose(got, legacy), est


def test_forward_value_preserved_under_correction():
    # Straight-through must keep the chosen estimator's forward VALUE (only the gradient changes).
    torch.manual_seed(42)
    log_probs = torch.randn(4, 8)
    log_probs_base = torch.randn(4, 8)
    for est in ESTIMATORS:
        legacy = compute_approx_kl(log_probs, log_probs_base, kl_estimator=est, unbiased_gradient=False)
        corrected = compute_approx_kl(log_probs, log_probs_base, kl_estimator=est, unbiased_gradient=True)
        assert torch.allclose(corrected, legacy), est


def test_gradients_equal_across_estimators_on_policy():
    # On-policy (log_probs_old=None -> w=1): k1/k2/k3 must share the SAME corrected gradient,
    # equal to the analytic reverse-KL gradient  k1 * d/dtheta log pi_theta = log_ratio * 1.
    torch.manual_seed(42)
    log_probs = torch.randn(4, 8)
    log_probs_base = torch.randn(4, 8)
    analytic = log_probs.float() - log_probs_base.float()  # k1 coefficient; d(log_probs)/d(log_probs)=1

    grads = [
        _grad_wrt_log_probs(log_probs, log_probs_base, kl_estimator=e, unbiased_gradient=True) for e in ESTIMATORS
    ]
    for g, est in zip(grads, ESTIMATORS):
        assert torch.allclose(g, analytic, atol=1e-5), est
    assert torch.allclose(grads[0], grads[1], atol=1e-6)
    assert torch.allclose(grads[1], grads[2], atol=1e-6)


def test_importance_weight_scales_gradient_off_policy():
    # Off-policy: gradient must scale by the clamped IS weight w = clamp(exp(log_probs - log_probs_old), max).
    torch.manual_seed(42)
    log_probs = torch.randn(4, 8)
    log_probs_base = torch.randn(4, 8)
    log_probs_old = log_probs - 0.3 * torch.randn(4, 8)  # theta != theta_old

    w = (log_probs.float() - log_probs_old.float()).exp().clamp(max=KL_IS_WEIGHT_CLIP)
    analytic = w * (log_probs.float() - log_probs_base.float())

    for est in ESTIMATORS:
        g = _grad_wrt_log_probs(
            log_probs, log_probs_base, log_probs_old=log_probs_old, kl_estimator=est, unbiased_gradient=True
        )
        assert torch.allclose(g, analytic, atol=1e-5), est


def test_k1_off_gradient_is_broken_baseline():
    # Sanity that the legacy path really is the biased one: k1 without correction has a constant
    # gradient (1 per element) that does NOT depend on the reference -> not a reverse-KL gradient.
    torch.manual_seed(42)
    log_probs = torch.randn(4, 8)
    log_probs_base = torch.randn(4, 8)
    g = _grad_wrt_log_probs(log_probs, log_probs_base, kl_estimator="k1", unbiased_gradient=False)
    assert torch.allclose(g, torch.ones_like(g))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("corrected", [False, True])
def test_k3_extreme_finite_log_probs_match_clipped_reference(dtype, corrected):
    log_probs = torch.tensor([-100.0, -90.0, -11.0, -10.0, -3.0, -1.0, -0.1, 0.0], dtype=dtype, requires_grad=True)
    base = torch.zeros_like(log_probs)
    actual = compute_approx_kl(log_probs, base, kl_estimator="k3", unbiased_gradient=corrected)
    actual.sum().backward()

    reference_input = log_probs.detach().double().requires_grad_(True)
    reference = ((-reference_input).exp() - 1 + reference_input).clamp(-10, 10)
    reference.sum().backward()
    expected_grad = reference_input.detach() if corrected else reference_input.grad
    torch.testing.assert_close(actual, reference.float(), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(log_probs.grad, expected_grad.to(dtype), atol=1e-6, rtol=1e-5)


def test_k3_kl_regularization_does_not_corrupt_optimizer_update():
    logits = torch.nn.Parameter(torch.tensor([[0.0, -100.0], [0.0, -1.0]]))
    reference_logits = torch.tensor([[0.0, 0.0], [0.0, -0.2]])
    optimizer = torch.optim.AdamW([logits], lr=1e-3)
    log_probs = logits.log_softmax(-1)[:, 1]
    base = reference_logits.log_softmax(-1)[:, 1]
    loss = compute_approx_kl(log_probs, base, kl_estimator="k3").mean() * 1e-3
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[1].abs().sum() > 0
    optimizer.step()
    assert torch.isfinite(logits).all()
