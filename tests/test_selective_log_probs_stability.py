import importlib.util
import math
from pathlib import Path

import deepspeed
import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

# Load the complete numerical module without importing the unrelated GPU Actor.
# DeepSpeed and Torch are their real installed packages, not import replacements.
assert Path(deepspeed.__file__).is_file()
_spec = importlib.util.spec_from_file_location(
    "openrlhf_native_log_probs", Path(__file__).resolve().parents[1] / "openrlhf/models/utils.py"
)
_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_utils)
log_probs_from_logits = _utils.log_probs_from_logits


@pytest.mark.parametrize("dtype,offset", [(torch.float32, 1e8), (torch.float64, 1e16)])
@pytest.mark.parametrize("sign", [-1, 0, 1])
def test_uniform_selected_log_probs_preserve_values_and_gradients(dtype, offset, sign):
    logits = torch.full((2, 3, 7), sign * offset, dtype=dtype, requires_grad=True)
    labels = torch.tensor([[0, 1, 6], [2, 5, 3]])
    actual = log_probs_from_logits(logits, labels)
    expected = torch.full((2, 3), -math.log(7), dtype=dtype)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    actual.sum().backward()
    expected_gradient = torch.full_like(logits, -1 / 7)
    expected_gradient.scatter_add_(-1, labels.unsqueeze(-1), torch.ones_like(labels, dtype=dtype).unsqueeze(-1))
    torch.testing.assert_close(logits.grad, expected_gradient, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("dtype,offset,spacing", [(torch.float32, 1e8, 8), (torch.float64, 1e16, 2)])
@pytest.mark.parametrize("sign", [-1, 1])
def test_nonuniform_noncontiguous_logits_match_observed_input_reference(dtype, offset, spacing, sign):
    logits = (torch.arange(48, dtype=dtype).reshape(2, 8, 3) * spacing + sign * offset).transpose(1, 2)
    logits = logits.detach().requires_grad_()
    assert not logits.is_contiguous()
    labels = torch.tensor([[0, 1, 2], [7, 4, 6]])
    before = logits.detach().clone()
    centered = before.double() - before.double().amax(dim=-1, keepdim=True)
    expected_log_probs = centered.log_softmax(-1)
    expected_gradient = -expected_log_probs.exp()
    expected_gradient.scatter_add_(-1, labels.unsqueeze(-1), torch.ones(2, 3, 1, dtype=torch.float64))
    actual = log_probs_from_logits(logits, labels)
    torch.testing.assert_close(actual, expected_log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1).to(dtype))
    actual.sum().backward()
    torch.testing.assert_close(logits.grad, expected_gradient.to(dtype), rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(logits.detach(), before, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("rows", [0, 1, 1023, 1024, 1025, 2051])
def test_selected_log_probs_preserve_chunk_boundaries_and_empty_gradients(dtype, rows):
    generator = torch.Generator().manual_seed(23)
    logits = torch.randn(rows, 9, dtype=dtype, generator=generator, requires_grad=True)
    labels = torch.arange(rows) % 9
    reference = logits.detach().clone().requires_grad_()
    expected = reference.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    actual = log_probs_from_logits(logits, labels)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(logits.grad, reference.grad, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0])
def test_existing_dtypes_and_temperature_scaling(dtype, temperature):
    original = torch.tensor([[[0.0, -2.0, 3.0], [1.0, 2.0, -1.0]]], dtype=dtype, requires_grad=True)
    logits = original.clone()
    labels = torch.tensor([[2, 0]])
    expected = (original / temperature).log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    actual = log_probs_from_logits(logits, labels, temperature=temperature)
    torch.testing.assert_close(actual, expected)
    actual_gradient = torch.autograd.grad(actual.sum(), original, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected.sum(), original)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)
    torch.testing.assert_close(logits, original / temperature)


@pytest.mark.parametrize("labels", [torch.tensor([-1]), torch.tensor([3])])
def test_selected_log_probs_preserve_invalid_label_errors(labels):
    with pytest.raises((RuntimeError, IndexError)):
        log_probs_from_logits(torch.zeros(1, 3), labels)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_selected_log_probs_keep_nonfinite_row_semantics(value):
    logits = torch.full((2, 3), value)
    assert torch.isnan(log_probs_from_logits(logits, torch.tensor([0, 2]))).all()


def test_backward_does_not_save_an_extra_full_vocabulary_buffer():
    logits = torch.randn(2051, 257, requires_grad=True)
    labels = torch.arange(2051) % 257
    original_pointers = {logits.untyped_storage().data_ptr(), labels.untyped_storage().data_ptr()}
    saved = {}

    def pack(tensor):
        storage = tensor.untyped_storage()
        if storage.data_ptr() not in original_pointers:
            saved[storage.data_ptr()] = storage.nbytes()
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        result = log_probs_from_logits(logits, labels)
    # Preserve the original fallback's row-sized saved-state budget, including
    # checkpoint bookkeeping; a retained vocabulary buffer is over 2 MB here.
    assert sum(saved.values()) < 2051 * 128, saved
    result.sum().backward()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("temperature", [0.5, 2.0])
@pytest.mark.parametrize("offset", [-1e8, 1e8])
def test_native_gpt2_updates_match_stable_loss_after_checkpoint_resume(tmp_path, temperature, offset):
    torch.manual_seed(17)
    config = GPT2Config(
        vocab_size=16,
        n_positions=8,
        n_embd=8,
        n_layer=1,
        n_head=2,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    model = GPT2LMHeadModel(config)
    reference = GPT2LMHeadModel(config)
    reference.load_state_dict(model.state_dict())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01, momentum=0.8)
    tokens = torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])
    labels = tokens[:, 1:]
    initial = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    for step in range(3):
        optimizer.zero_grad()
        reference_optimizer.zero_grad()
        observed_logits = model(tokens).logits[:, :-1] + offset
        reference_logits = (reference(tokens).logits[:, :-1] + offset) / temperature
        actual = log_probs_from_logits(observed_logits, labels, temperature=temperature)
        expected = reference_logits.double().log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        torch.testing.assert_close(actual, expected.to(actual.dtype), rtol=1e-6, atol=1e-7)
        (-actual.mean()).backward()
        (-expected.mean()).backward()
        for parameter, ref_parameter in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(parameter.grad, ref_parameter.grad, rtol=1e-5, atol=1e-6)
        optimizer.step()
        reference_optimizer.step()
        if step == 1:
            path = tmp_path / "checkpoint.pt"
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, path)
            restored = torch.load(path, weights_only=True)
            model = GPT2LMHeadModel(config)
            model.load_state_dict(restored["model"])
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
            optimizer.load_state_dict(restored["optimizer"])
    assert any(not torch.equal(parameter, initial[name]) for name, parameter in model.named_parameters())
    for parameter, ref_parameter in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(parameter, ref_parameter, rtol=1e-5, atol=1e-6)
