import importlib.machinery
import importlib.util
import math
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutputWithPast

_TEST_PACKAGE = "_openrlhf_actor_test"


def _load_actor_module():
    root = Path(__file__).resolve().parents[1]
    models_dir = root / "openrlhf" / "models"

    # Keep this CPU test importable without DeepSpeed, flash-attn, or peft.
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
            sys.modules[name] = stub
            inserted_stubs.append(name)

    pkg = types.ModuleType(_TEST_PACKAGE)
    pkg.__path__ = [str(models_dir)]
    sys.modules[_TEST_PACKAGE] = pkg

    # compute_entropy is wrapped in torch.compile at import time; run it eagerly here.
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
        # Drop the import-time stubs so later tests can import the real packages
        # (a lingering MagicMock "deepspeed" breaks test_deepspeed_save_model).
        for name in inserted_stubs:
            del sys.modules[name]

    return sys.modules[f"{_TEST_PACKAGE}.actor"]


Actor = _load_actor_module().Actor


class _TableLM(torch.nn.Module):
    """Logits are a fixed per-token table so the expected values are easy to recompute."""

    def __init__(self, vocab_size=11):
        super().__init__()
        generator = torch.Generator().manual_seed(0)
        self.table = torch.nn.Parameter(torch.randn(vocab_size, vocab_size, generator=generator))

    def forward(self, input_ids, attention_mask=None, position_ids=None, **_):
        return CausalLMOutputWithPast(logits=self.table[input_ids])


@pytest.mark.parametrize("temperature", [0.5, 1.5])
def test_entropy_matches_temperature_scaled_policy(temperature):
    actor = Actor(_TableLM(), temperature=temperature)
    actor.packing_samples = False

    generator = torch.Generator().manual_seed(1)
    sequences = torch.randint(0, 11, (2, 6), generator=generator)
    attention_mask = torch.ones_like(sequences)
    action_mask = torch.ones(2, 5, dtype=torch.long)

    action_log_probs, output = actor(
        sequences, action_mask, attention_mask=attention_mask, return_output=True, return_entropy=True
    )

    policy_logits = _TableLM()(sequences).logits / temperature
    next_tokens = torch.roll(sequences, shifts=-1, dims=1).unsqueeze(-1)
    expected_log_probs = torch.log_softmax(policy_logits, dim=-1).gather(-1, next_tokens).squeeze(-1)[:, :-1]
    expected_entropy = torch.distributions.Categorical(logits=policy_logits).entropy()[:, :-1]

    assert torch.allclose(action_log_probs, expected_log_probs, atol=1e-5)
    # The entropy bonus must describe the same policy the log probs (and the sampler) use.
    assert torch.allclose(output.entropy, expected_entropy, atol=1e-5)


compute_entropy = sys.modules[f"{_TEST_PACKAGE}.utils"].compute_entropy


def _entropy_reference(logits):
    # Evaluate the observed input in double precision after removing its common offset.
    centered = logits.detach().double() - logits.detach().double().amax(dim=-1, keepdim=True)
    log_probs = centered.log_softmax(dim=-1)
    probabilities = log_probs.exp()
    entropy = -(probabilities * log_probs.masked_fill(probabilities == 0, 0)).sum(dim=-1)
    gradient = -probabilities * (log_probs.masked_fill(probabilities == 0, 0) + entropy.unsqueeze(-1))
    return entropy.to(logits.dtype), gradient.to(logits.dtype)


_ENTROPY_DTYPES = [
    (torch.float32, 1e7, 3e-6, 3e-7),
    (torch.float64, 1e16, 1e-12, 1e-12),
    (torch.float16, 1000, 3e-3, 5e-4),
    (torch.bfloat16, 100, 2e-2, 4e-3),
]


@pytest.mark.parametrize("dtype,offset,rtol,atol", _ENTROPY_DTYPES)
@pytest.mark.parametrize("sign", [0, -1, 1])
def test_uniform_entropy_and_gradient_ignore_common_logit_offsets(dtype, offset, rtol, atol, sign):
    logits = torch.full((2, 3, 4), sign * offset, dtype=dtype, requires_grad=True)
    entropy = compute_entropy(logits)
    entropy.sum().backward()
    torch.testing.assert_close(entropy, torch.full_like(entropy, math.log(4)), rtol=rtol, atol=atol)
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits), rtol=0, atol=atol)


@pytest.mark.parametrize("dtype,offset,rtol,atol", _ENTROPY_DTYPES)
@pytest.mark.parametrize("sign", [-1, 1])
def test_nonuniform_entropy_matches_probability_domain_value_and_gradient(dtype, offset, rtol, atol, sign):
    logits = (torch.arange(4, dtype=dtype).expand(2, 4) + sign * offset).clone().requires_grad_()
    original = logits.detach().clone()
    expected_entropy, expected_gradient = _entropy_reference(logits)
    entropy = compute_entropy(logits)
    entropy.sum().backward()
    torch.testing.assert_close(entropy, expected_entropy, rtol=rtol, atol=atol)
    torch.testing.assert_close(logits.grad, expected_gradient, rtol=rtol, atol=atol)
    torch.testing.assert_close(logits.detach(), original, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
def test_entropy_keeps_saturated_finite_distributions_finite(dtype):
    largest = torch.finfo(dtype).max
    logits = torch.tensor([[largest, -largest, 0.0, -largest]], dtype=dtype, requires_grad=True)
    entropy = compute_entropy(logits)
    entropy.sum().backward()
    torch.testing.assert_close(entropy, torch.zeros_like(entropy), rtol=0, atol=0)
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits), rtol=0, atol=0)


def test_entropy_preserves_nan_inputs():
    logits = torch.tensor([[0.0, float("nan"), 1.0]])
    assert torch.isnan(compute_entropy(logits)).all()


@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("offset", [-1e8, 1e8])
def test_actor_uniform_entropy_bonus_does_not_update_gpt2(temperature, offset):
    torch.manual_seed(13)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=16,
            n_positions=8,
            n_embd=8,
            n_layer=1,
            n_head=2,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )
    )
    with torch.no_grad():
        model.lm_head.weight.zero_()

    def shift_logits(module, inputs, output):
        output.logits = output.logits + offset
        return output

    handle = model.register_forward_hook(shift_logits)
    actor = Actor(model, temperature=temperature)
    actor.packing_samples = False
    initial = {name: parameter.detach().clone() for name, parameter in actor.named_parameters()}
    optimizer = torch.optim.SGD(actor.parameters(), lr=0.1, momentum=0.8)
    sequences = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    for _ in range(2):
        optimizer.zero_grad()
        output = actor(sequences, attention_mask=torch.ones_like(sequences), return_output=True, return_entropy=True)
        torch.testing.assert_close(output.entropy, torch.full((2, 3), math.log(16)), rtol=1e-6, atol=1e-6)
        (-0.01 * output.entropy.mean()).backward()
        for parameter in actor.parameters():
            if parameter.grad is not None:
                torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter), rtol=0, atol=1e-8)
        optimizer.step()
    for name, parameter in actor.named_parameters():
        torch.testing.assert_close(parameter, initial[name], rtol=0, atol=1e-8)
    handle.remove()


@pytest.mark.parametrize("dtype,offset", [(torch.float32, 1e7), (torch.float64, 1e16)])
def test_compiled_entropy_matches_probability_domain_reference(dtype, offset):
    compiled = torch.compile(compute_entropy, fullgraph=True)
    logits = (torch.arange(4, dtype=dtype).expand(2, 4) + offset).clone().requires_grad_()
    expected_entropy, expected_gradient = _entropy_reference(logits)
    entropy = compiled(logits)
    entropy.sum().backward()
    torch.testing.assert_close(entropy, expected_entropy, rtol=3e-6, atol=3e-7)
    torch.testing.assert_close(logits.grad, expected_gradient, rtol=3e-6, atol=3e-7)
