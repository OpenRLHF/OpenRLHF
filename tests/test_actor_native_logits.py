import importlib
import importlib.machinery
import importlib.util
import math
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast

_TEST_PACKAGE = "_openrlhf_native_logits_test"


def _load_actor_module():
    root = Path(__file__).resolve().parents[1]
    models_dir = root / "openrlhf" / "models"

    # Keep these tests importable without DeepSpeed, flash-attn, or peft.
    inserted_stubs = []
    for name in (
        "deepspeed",
        "flash_attn",
        "flash_attn.bert_padding",
        "flash_attn.utils",
        "flash_attn.utils.distributed",
        "peft",
        "peft.tuners",
        "peft.tuners.lora",
    ):
        if name not in sys.modules:
            stub = MagicMock()
            stub.__spec__ = importlib.machinery.ModuleSpec(name, None)
            if name == "peft":
                stub.LoraConfig = object
                stub.TaskType = types.SimpleNamespace(CAUSAL_LM="CAUSAL_LM")
                stub.get_peft_model = lambda model, _: model
            if name == "peft.tuners.lora":
                stub.LoraLayer = object
            sys.modules[name] = stub
            inserted_stubs.append(name)

    package = types.ModuleType(_TEST_PACKAGE)
    package.__path__ = [str(models_dir)]
    sys.modules[_TEST_PACKAGE] = package

    # compute_entropy is compiled at import time; eager execution keeps this CPU test portable.
    original_compile = torch.compile
    torch.compile = lambda fn=None, **_: fn if fn is not None else (lambda f: f)
    try:
        for name in ("utils", "ring_attn_utils", "actor"):
            spec = importlib.util.spec_from_file_location(f"{_TEST_PACKAGE}.{name}", models_dir / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[f"{_TEST_PACKAGE}.{name}"] = module
            spec.loader.exec_module(module)
    finally:
        torch.compile = original_compile
        for name in inserted_stubs:
            sys.modules.pop(name, None)

    return sys.modules[f"{_TEST_PACKAGE}.actor"]


Actor = _load_actor_module().Actor


class _TableLM(torch.nn.Module):
    def __init__(self, vocab_size=13, dtype=torch.float32, aux_loss=None):
        super().__init__()
        generator = torch.Generator().manual_seed(7)
        table = torch.randn(vocab_size, vocab_size, generator=generator, dtype=dtype)
        self.table = torch.nn.Parameter(table)
        self.aux_loss = aux_loss
        self.forward_calls = 0

    def forward(self, input_ids, attention_mask=None, position_ids=None, **_):
        self.forward_calls += 1
        output = CausalLMOutputWithPast(logits=self.table[input_ids])
        if self.aux_loss is not None:
            output["aux_loss"] = self.aux_loss
        return output


def _actor(model, temperature=1.0):
    actor = Actor(model, temperature=temperature)
    actor.packing_samples = False
    return actor


def _inputs(device="cpu"):
    sequences = torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]], device=device)
    attention_mask = torch.ones_like(sequences)
    action_mask = torch.tensor([[1, 0, 1], [1, 1, 0]], device=device)
    return sequences, attention_mask, action_mask


def _reference_log_probs(logits, sequences, temperature):
    labels = torch.roll(sequences, shifts=-1, dims=1)
    scaled = logits.float() / temperature
    return F.log_softmax(scaled, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)[:, :-1]


@pytest.mark.parametrize("temperature", [0.5, 1.0, 1.5])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cpu_fallback_matches_fp32_forward_and_backward_without_mutating_inputs(temperature, dtype):
    model = _TableLM(dtype=dtype)
    actor = _actor(model, temperature)
    sequences, attention_mask, action_mask = _inputs()
    snapshots = tuple(t.clone() for t in (sequences, attention_mask, action_mask))

    action_log_probs = actor(sequences, action_mask, attention_mask=attention_mask)
    actor_loss = action_log_probs.sum()
    actor_loss.backward()
    actual_grad = model.table.grad.detach().clone()

    expected = _reference_log_probs(model.table[sequences], sequences, temperature)
    expected = expected[:, -action_mask.shape[1] :] * action_mask.float()
    expected_grad = torch.autograd.grad(expected.sum(), model.table)[0]

    assert action_log_probs.dtype == torch.float32
    tolerance = 1e-6 if dtype == torch.float32 else 2e-3
    assert torch.allclose(action_log_probs, expected, atol=tolerance, rtol=tolerance)
    assert torch.allclose(actual_grad, expected_grad, atol=tolerance, rtol=tolerance)
    for actual, snapshot in zip((sequences, attention_mask, action_mask), snapshots):
        assert torch.equal(actual, snapshot)


def test_cpu_equal_logits_bfloat16_returns_exact_uniform_logprobs_and_reference_gradients():
    vocab_size = 17
    model = _TableLM(vocab_size=vocab_size, dtype=torch.bfloat16)
    with torch.no_grad():
        model.table.fill_(1.25)
    actor = _actor(model, temperature=1.5)
    sequences, attention_mask, action_mask = _inputs()
    action_mask = torch.ones_like(action_mask)

    actual, output = actor(
        sequences,
        action_mask,
        attention_mask=attention_mask,
        return_output=True,
        return_logits=False,
    )
    actual.sum().backward()
    actual_grad = model.table.grad.detach().float()

    expected = torch.full_like(actual, -math.log(vocab_size))
    reference = _reference_log_probs(model.table[sequences], sequences, 1.5)
    reference = reference[:, -action_mask.shape[1] :] * action_mask.float()
    reference_grad = torch.autograd.grad(reference.sum(), model.table)[0].float()

    assert actual.dtype == torch.float32
    assert output.logits is None
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    assert torch.allclose(actual, reference, atol=2e-3, rtol=2e-3)
    assert torch.allclose(actual_grad, reference_grad, atol=2e-3, rtol=2e-3)


def test_default_returned_logits_are_fp32_and_temperature_scaled():
    model = _TableLM()
    actor = _actor(model, temperature=1.5)
    sequences, attention_mask, _ = _inputs()

    output = actor(sequences, attention_mask=attention_mask, return_output=True)

    expected = model.table[sequences].float() / 1.5
    assert output.logits.dtype == torch.float32
    assert torch.allclose(output.logits, expected, atol=1e-6, rtol=1e-6)


def test_return_logits_false_hides_logits_but_preserves_aux_loss():
    aux_loss = torch.tensor(0.375)
    model = _TableLM(aux_loss=aux_loss)
    actor = _actor(model)
    sequences, attention_mask, action_mask = _inputs()

    _, output = actor(
        sequences,
        action_mask,
        attention_mask=attention_mask,
        return_output=True,
        return_logits=False,
    )

    assert output.logits is None
    assert output.aux_loss is aux_loss


def test_entropy_forces_fp32_fallback_and_uses_scaled_logits():
    model = _TableLM(dtype=torch.float16)
    actor = _actor(model, temperature=0.5)
    sequences, attention_mask, action_mask = _inputs()

    _, output = actor(
        sequences,
        action_mask,
        attention_mask=attention_mask,
        return_output=True,
        return_entropy=True,
    )

    scaled = model.table[sequences].float() / 0.5
    expected_entropy = torch.logsumexp(scaled, dim=-1) - (scaled.softmax(dim=-1) * scaled).sum(dim=-1)
    assert output.logits.dtype == torch.float32
    assert torch.allclose(output.logits, scaled, atol=1e-3, rtol=1e-3)
    assert torch.allclose(output.entropy, expected_entropy[:, :-1], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("use_action_mask", [True, False])
def test_action_mask_and_return_logprobs_paths_match_reference(use_action_mask):
    model = _TableLM()
    actor = _actor(model, temperature=1.5)
    sequences, attention_mask, action_mask = _inputs()
    kwargs = {"action_mask": action_mask} if use_action_mask else {"return_logprobs": True}

    actual = actor(sequences, attention_mask=attention_mask, **kwargs)
    expected = _reference_log_probs(model.table[sequences], sequences, 1.5)
    if use_action_mask:
        expected = expected[:, -action_mask.shape[1] :] * action_mask.float()

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_return_logits_false_without_logprobs_fails_before_model_call():
    model = _TableLM()
    actor = _actor(model)
    sequences, attention_mask, _ = _inputs()

    with pytest.raises(ValueError):
        actor(sequences, attention_mask=attention_mask, return_output=True, return_logits=False)

    assert model.forward_calls == 0


def _flash_cross_entropy_module():
    if not torch.cuda.is_available():
        return None
    try:
        module = importlib.import_module("flash_attn.ops.triton.cross_entropy")
    except (ImportError, ModuleNotFoundError):
        return None
    return module if hasattr(module, "cross_entropy_loss") else None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("return_output", [False, True])
@pytest.mark.parametrize(
    ("dtype", "gradient_tolerance"),
    [(torch.float16, 5e-3), (torch.bfloat16, 2e-2)],
)
def test_cuda_native_cross_entropy_matches_values_gradients_and_scale(
    monkeypatch, return_output, dtype, gradient_tolerance
):
    cross_entropy = _flash_cross_entropy_module()
    if cross_entropy is None:
        pytest.skip("flash-attn cross entropy is unavailable")

    calls = []
    real_kernel = cross_entropy.cross_entropy_loss

    def wrapped_kernel(*args, **kwargs):
        logits = kwargs["logits"] if "logits" in kwargs else args[0]
        scale = kwargs.get("logit_scale", args[2] if len(args) > 2 else 1.0)
        calls.append((logits.dtype, float(scale), logits.device.type))
        return real_kernel(*args, **kwargs)

    monkeypatch.setattr(cross_entropy, "cross_entropy_loss", wrapped_kernel)
    actor_module = sys.modules[f"{_TEST_PACKAGE}.actor"]
    if hasattr(actor_module, "cross_entropy_loss"):
        monkeypatch.setattr(actor_module, "cross_entropy_loss", wrapped_kernel)

    model = _TableLM(dtype=dtype).cuda()
    aux_loss = torch.tensor(0.375, device="cuda") if return_output else None
    model.aux_loss = aux_loss
    actor = _actor(model, temperature=1.5)
    sequences, attention_mask, action_mask = _inputs("cuda")

    result = actor(
        sequences,
        action_mask,
        attention_mask=attention_mask,
        return_output=return_output,
        return_logits=not return_output,
    )
    if return_output:
        actual, output = result
        assert output.logits is None
        assert output.aux_loss is aux_loss
    else:
        actual = result
    assert actual.dtype == torch.float32
    actual.sum().backward()
    actual_grad = model.table.grad.detach().float().clone()

    expected = _reference_log_probs(model.table[sequences], sequences, 1.5)
    expected = expected[:, -action_mask.shape[1] :] * action_mask.float()
    expected_grad = torch.autograd.grad(expected.sum(), model.table)[0].float()

    assert calls
    assert all(call_dtype == dtype and device == "cuda" for call_dtype, _, device in calls)
    assert all(scale == pytest.approx(1 / 1.5) for _, scale, _ in calls)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
    assert torch.allclose(actual_grad, expected_grad, atol=gradient_tolerance, rtol=gradient_tolerance)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_default_output_is_fp32_and_does_not_use_native_kernel(monkeypatch, dtype):
    cross_entropy = _flash_cross_entropy_module()
    if cross_entropy is None:
        pytest.skip("flash-attn cross entropy is unavailable")

    calls = []
    real_kernel = cross_entropy.cross_entropy_loss

    def wrapped_kernel(*args, **kwargs):
        logits = kwargs["logits"] if "logits" in kwargs else args[0]
        scale = kwargs.get("logit_scale", args[2] if len(args) > 2 else 1.0)
        calls.append((logits.dtype, float(scale)))
        return real_kernel(*args, **kwargs)

    monkeypatch.setattr(cross_entropy, "cross_entropy_loss", wrapped_kernel)
    actor_module = sys.modules[f"{_TEST_PACKAGE}.actor"]
    if hasattr(actor_module, "cross_entropy_loss"):
        monkeypatch.setattr(actor_module, "cross_entropy_loss", wrapped_kernel)

    model = _TableLM(dtype=dtype).cuda()
    actor = _actor(model, temperature=0.5)
    sequences, attention_mask, action_mask = _inputs("cuda")

    _, output = actor(
        sequences,
        action_mask,
        attention_mask=attention_mask,
        return_output=True,
    )

    expected = model.table[sequences].float() / 0.5
    # return_output=True with the default return_logits=True keeps the original
    # FP32 output path; any utility-kernel call is therefore already FP32.
    assert all(dtype == torch.float32 and scale == pytest.approx(1.0) for dtype, scale in calls)
    assert output.logits.dtype == torch.float32
    assert torch.allclose(output.logits, expected, atol=1e-5, rtol=1e-5)
