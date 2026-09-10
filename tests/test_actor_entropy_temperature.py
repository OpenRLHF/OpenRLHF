import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
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
