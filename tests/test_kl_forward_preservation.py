import importlib.util
import math
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

_spec = importlib.util.spec_from_file_location(
    "_openrlhf_kl_forward_utils", Path(__file__).resolve().parents[1] / "openrlhf" / "models" / "utils.py"
)
_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_utils)
compute_approx_kl = _utils.compute_approx_kl


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("estimator", ["k1", "k2", "k3"])
@pytest.mark.parametrize("off_policy", [False, True])
def test_corrected_forward_preserves_values_with_large_finite_surrogates(dtype, estimator, off_policy):
    current = torch.tensor([-12000.0, -10.0, -0.25, -3.0, -50.0], dtype=dtype, requires_grad=True)
    reference = torch.tensor([-1.0, -14000.0, -0.125, -3.0, -1.0], dtype=dtype)
    previous = current.detach() - math.log(5) if off_policy else None
    expected = compute_approx_kl(current.detach(), reference, kl_estimator=estimator)
    actual = compute_approx_kl(
        current, reference, log_probs_old=previous, kl_estimator=estimator, unbiased_gradient=True
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    upstream = torch.tensor([0.001, 0.002, -0.003, 0.004, 0.005])
    actual.backward(upstream)
    weight = (
        (current.detach().float() - previous.float()).exp().clamp(max=_utils.KL_IS_WEIGHT_CLIP) if off_policy else 1.0
    )
    expected_gradient = upstream * weight * (current.detach().float() - reference.float())
    torch.testing.assert_close(current.grad, expected_gradient.to(dtype), rtol=1e-5, atol=1e-6)
    assert torch.isfinite(current.grad).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("estimator,expected_value", [("k1", -10.0), ("k2", 10.0), ("k3", 10.0)])
def test_corrected_forward_preserves_clipped_values_for_normalized_categorical_policies(
    dtype, estimator, expected_value
):
    logits = torch.tensor([[0.0, -10000.0], [0.0, -6000.0]], dtype=dtype, requires_grad=True)
    reference_logits = torch.tensor([[0.0, -1.0], [0.0, -1.0]], dtype=dtype)
    old_logits = torch.tensor([[0.0, -10002.0], [0.0, -6002.0]], dtype=dtype)
    log_probs = logits.log_softmax(-1)[:, 1]
    actual = compute_approx_kl(
        log_probs,
        reference_logits.log_softmax(-1)[:, 1],
        log_probs_old=old_logits.log_softmax(-1)[:, 1],
        kl_estimator=estimator,
        unbiased_gradient=True,
    )
    torch.testing.assert_close(actual, torch.full((2,), expected_value), rtol=0, atol=0)
    (actual.mean() * 1e-4).backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad) == logits.numel()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("optimizer_type", [torch.optim.SGD, torch.optim.AdamW])
@pytest.mark.parametrize("estimator", ["k1", "k2", "k3"])
def test_corrected_kl_keeps_reported_loss_and_analytic_training_updates(tmp_path, dtype, optimizer_type, estimator):
    torch.manual_seed(31)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=17,
            n_positions=8,
            n_embd=8,
            n_layer=1,
            n_head=2,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
    ).to(dtype)
    reference = deepcopy(model)
    optimizer_args = {"lr": 1e-4, "momentum": 0.5} if optimizer_type is torch.optim.SGD else {"lr": 1e-4}
    optimizer = optimizer_type(model.parameters(), **optimizer_args)
    reference_optimizer = optimizer_type(reference.parameters(), **optimizer_args)
    tokens = torch.tensor([[2, 3, 5, 7, 11], [3, 5, 7, 9, 13]])
    labels = tokens[:, 1:]
    with torch.no_grad():
        initial_logits = reference(tokens).logits[:, :-1]
        teacher_logits = initial_logits.clone()
        teacher_logits.scatter_add_(
            -1, labels.unsqueeze(-1), torch.full_like(labels.unsqueeze(-1), -12000, dtype=dtype)
        )
        old_logits = initial_logits.clone()
        old_logits.scatter_add_(-1, labels.unsqueeze(-1), torch.full_like(labels.unsqueeze(-1), -2, dtype=dtype))
        base_log_probs = teacher_logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        old_log_probs = old_logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    for step in range(3):
        optimizer.zero_grad()
        reference_optimizer.zero_grad()
        log_probs = model(tokens).logits[:, :-1].log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        reference_log_probs = (
            reference(tokens).logits[:, :-1].log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        )
        divergence = compute_approx_kl(
            log_probs,
            base_log_probs,
            log_probs_old=old_log_probs,
            kl_estimator=estimator,
            unbiased_gradient=True,
        )
        torch.testing.assert_close(divergence, torch.full_like(divergence, 10), rtol=0, atol=0)
        loss = -log_probs.mean() + 1e-4 * divergence.mean()
        expected_loss = -reference_log_probs.mean() + 1e-4 * torch.tensor(10.0)
        torch.testing.assert_close(loss, expected_loss, rtol=1e-6, atol=1e-7)
        loss.backward()

        # Supply the analytic reverse-KL gradient directly, independent of the straight-through expression.
        weight = (reference_log_probs.detach().float() - old_log_probs.float()).exp().clamp(max=10)
        ratio = reference_log_probs.detach().float() - base_log_probs.float()
        expected_gradient = weight * ratio * (1e-4 / reference_log_probs.numel())
        (-reference_log_probs.mean()).backward(retain_graph=True)
        reference_log_probs.backward(expected_gradient.to(dtype))
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-5, atol=1e-6)
        optimizer.step()
        reference_optimizer.step()
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        actual_state, expected_state = optimizer.state_dict(), reference_optimizer.state_dict()
        assert actual_state["param_groups"] == expected_state["param_groups"]
        for index, state in actual_state["state"].items():
            for name, value in state.items():
                torch.testing.assert_close(value, expected_state["state"][index][name], rtol=1e-5, atol=1e-6)
        if step == 0:
            checkpoint_path = tmp_path / "kl-training.pt"
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint_path)
            model = deepcopy(model)
            optimizer = optimizer_type(model.parameters(), **optimizer_args)
            checkpoint = torch.load(checkpoint_path, weights_only=True)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])


def test_nonfinite_inputs_are_not_hidden_by_forward_preservation():
    current = torch.tensor([float("nan"), -3.0], requires_grad=True)
    reference = torch.tensor([-1.0, -2.0])
    result = compute_approx_kl(current, reference, unbiased_gradient=True)
    assert torch.isnan(result[0])
    assert result[1] == -1


def test_uncorrected_forward_and_invalid_estimator_contracts_remain_unchanged():
    current = torch.tensor([-12000.0, -1.0], requires_grad=True)
    reference = torch.tensor([-1.0, -12000.0])
    result = compute_approx_kl(current, reference)
    torch.testing.assert_close(result, torch.tensor([-10.0, 10.0]), rtol=0, atol=0)
    result.sum().backward()
    torch.testing.assert_close(current.grad, torch.zeros_like(current), rtol=0, atol=0)
    with pytest.raises(ValueError, match="Unknown kl_estimator"):
        compute_approx_kl(current, reference, kl_estimator="unsupported", unbiased_gradient=True)
