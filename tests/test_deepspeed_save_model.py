"""Regression tests for DeepspeedStrategy.save_model's tie_word_embeddings
corner case (issue #747).

ZeRO-3's consolidated state dict omits the tied lm_head weight (it shares
storage with the token embedding), so save_model special-cases it out of the
"missing key" assertion. That special case only matched the bare
"lm_head.weight" key. PEFT wraps every parameter name with a
"base_model.model." prefix, so a LoRA-wrapped save of a tied-embedding model
(e.g. Qwen2-0.5B/1.5B/3B) always tripped the assertion.

These tests exercise the real save_model code path with a PEFT-wrapped tiny
model and a stand-in for DeepSpeed's ZeRO-3 consolidation, mocking only the
CUDA-only flash_attn import and the distributed barrier so it runs single
process on CPU.
"""

import importlib.machinery
import sys
import tempfile
import types
from types import SimpleNamespace

import pytest
import torch.nn as nn

peft = pytest.importorskip("peft")
pytest.importorskip("deepspeed")
pytest.importorskip("torchdata")


def _make_stub(name):
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    mod.__path__ = []
    return mod


@pytest.fixture
def ds_module(monkeypatch):
    """Import openrlhf.utils.deepspeed.deepspeed with flash_attn stubbed out.

    flash_attn only ships CUDA builds, but it is merely imported (never
    called) for the code path under test, so a bare stub is enough to make
    the module importable on a CPU-only machine.
    """
    flash_attn_mod = _make_stub("flash_attn")
    bert_padding_mod = _make_stub("flash_attn.bert_padding")
    for fn in ("index_first_axis", "pad_input", "rearrange", "unpad_input"):
        setattr(bert_padding_mod, fn, lambda *a, **k: None)
    utils_pkg = _make_stub("flash_attn.utils")
    distributed_mod = _make_stub("flash_attn.utils.distributed")
    distributed_mod.all_gather = lambda *a, **k: None
    utils_pkg.distributed = distributed_mod
    flash_attn_mod.bert_padding = bert_padding_mod
    flash_attn_mod.utils = utils_pkg

    for name, mod in (
        ("flash_attn", flash_attn_mod),
        ("flash_attn.bert_padding", bert_padding_mod),
        ("flash_attn.utils", utils_pkg),
        ("flash_attn.utils.distributed", distributed_mod),
    ):
        monkeypatch.setitem(sys.modules, name, mod)

    import openrlhf.utils.deepspeed.deepspeed as module

    # Single-process test: no real process group to barrier/sync on.
    monkeypatch.setattr(module, "torch_dist_barrier_and_cuda_sync", lambda: None)
    return module


class _TinyConfig:
    tie_word_embeddings = True

    def to_json_file(self, path):
        with open(path, "w") as f:
            f.write("{}")


class _TinyModel(nn.Module):
    """Minimal tied-embedding-shaped model: a real lm_head param plus a
    same-named submodule, so PEFT's "base_model.model." prefix and DeepSpeed's
    tied-weight dedup both apply the way they do to Qwen2/Qwen2.5-0.5B-3B."""

    def __init__(self):
        super().__init__()
        self.config = _TinyConfig()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(10, 4)
        self.q_proj = nn.Linear(4, 4)
        self.lm_head = nn.Linear(4, 10, bias=False)

    def forward(self, x):
        return self.lm_head(self.q_proj(self.model.embed_tokens(x)))


def _strategy(ds_module, zero_stage=3):
    strategy = object.__new__(ds_module.DeepspeedStrategy)
    strategy.args = SimpleNamespace(ds=SimpleNamespace(zero_stage=zero_stage, tensor_parallel_size=1))
    strategy.ds_tensor_parallel_size = 1
    strategy.stage = zero_stage
    return strategy


def _tokenizer():
    return SimpleNamespace(save_pretrained=lambda *a, **k: None)


def test_save_model_lora_tied_embeddings_zero3(ds_module):
    """PEFT-prefixed tied lm_head key must not trip the mismatch assertion."""
    base = _TinyModel()
    peft_model = peft.get_peft_model(base, peft.LoraConfig(target_modules=["q_proj"]))
    peft_model.save_pretrained = lambda *a, **k: None

    tied_key = "base_model.model.lm_head.weight"
    full_state = peft_model.state_dict()
    assert tied_key in full_state

    # Stand-in for DeepSpeed ZeRO-3's _consolidated_16bit_state_dict(), which
    # omits the tied lm_head weight because it shares storage with embed_tokens.
    consolidated = {k: v for k, v in full_state.items() if k != tied_key}
    peft_model._consolidated_16bit_state_dict = lambda: consolidated

    strategy = _strategy(ds_module)
    with tempfile.TemporaryDirectory() as tmp:
        strategy.save_model(peft_model, _tokenizer(), tmp)  # must not raise


def test_save_model_plain_tied_embeddings_zero3(ds_module):
    """Original (non-PEFT) tie_word_embeddings corner case keeps working."""
    model = _TinyModel()
    model.save_pretrained = lambda *a, **k: None

    tied_key = "lm_head.weight"
    full_state = model.state_dict()
    assert tied_key in full_state
    consolidated = {k: v for k, v in full_state.items() if k != tied_key}
    model._consolidated_16bit_state_dict = lambda: consolidated

    strategy = _strategy(ds_module)
    with tempfile.TemporaryDirectory() as tmp:
        strategy.save_model(model, _tokenizer(), tmp)  # must not raise


def test_save_model_genuine_mismatch_still_raises(ds_module):
    """A real (non-lm_head) missing key must still fail loudly."""
    model = _TinyModel()
    model.save_pretrained = lambda *a, **k: None

    full_state = model.state_dict()
    consolidated = {k: v for k, v in full_state.items() if k != "q_proj.weight"}
    model._consolidated_16bit_state_dict = lambda: consolidated

    strategy = _strategy(ds_module)
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(AssertionError, match="q_proj.weight"):
            strategy.save_model(model, _tokenizer(), tmp)
