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


@pytest.fixture(params=["llama", "qwen2"])
def tiny_lm(request, tmp_path):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    config_cls, model_cls = (
        (LlamaConfig, LlamaForCausalLM) if request.param == "llama" else (Qwen2Config, Qwen2ForCausalLM)
    )
    torch.manual_seed(17)
    config = config_cls(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        pad_token_id=0,
    )
    base_path = tmp_path / "base"
    model_cls(config).save_pretrained(base_path)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "yes": 2, "no": 3}, unk_token="[UNK]")),
        pad_token="[PAD]",
        unk_token="[UNK]",
    )
    tokenizer.save_pretrained(base_path)
    return base_path, tokenizer


@pytest.mark.parametrize(
    "targets,head_name,stage",
    [
        ("all-linear", "score", 2),
        (["q_proj", "v_proj"], "value_head", 2),
        ("all-linear", "value_head", 3),
        (["q_proj", "v_proj"], "score", 3),
    ],
)
@pytest.mark.parametrize("param_dtype", ["bf16", "fp16"])
def test_reward_lora_checkpoint_roundtrip(ds_module, tiny_lm, tmp_path, targets, head_name, stage, param_dtype):
    import torch
    from transformers import AutoConfig

    from openrlhf.cli.lora_combiner import apply_lora
    from openrlhf.models import get_llm_for_sequence_regression

    base_path, tokenizer = tiny_lm
    model = get_llm_for_sequence_regression(
        str(base_path),
        "reward",
        lora_rank=2,
        param_dtype=param_dtype,
        target_modules=targets,
        value_head_prefix=head_name,
        init_value_head=True,
        attn_implementation="eager",
    )
    head = getattr(model.base_model.model, head_name)
    assert head.weight.requires_grad
    initial_head = head.weight.detach().clone()
    input_ids = torch.tensor([[0, 0, 2, 3], [2, 3, 4, 5]])
    mask = input_ids.ne(0)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01)
    model.train()
    for _ in range(2):
        rewards = model(input_ids, attention_mask=mask)
        loss = -torch.nn.functional.logsigmoid(rewards[0] - rewards[1])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    model.eval()
    assert not torch.equal(initial_head, head.weight)
    # The trainer stores normalization statistics in config, not persistent buffers.
    model.config.mean, model.config.std = 0.25, 1.75
    expected = model(input_ids, attention_mask=mask).detach()
    adapter_path, merged_path = tmp_path / "adapter", tmp_path / "merged"
    strategy = _strategy(ds_module, zero_stage=stage)
    if stage == 3:
        # Exercise the real consolidated-state dispatch without a GPU ZeRO engine.
        model._consolidated_16bit_state_dict = lambda: model.state_dict()
    strategy.save_model(model, tokenizer, str(adapter_path))
    assert AutoConfig.from_pretrained(adapter_path).value_head_prefix == head_name
    restored = get_llm_for_sequence_regression(
        str(base_path),
        "reward",
        config=AutoConfig.from_pretrained(adapter_path),
        param_dtype=param_dtype,
        attn_implementation="eager",
    )
    restored = peft.PeftModel.from_pretrained(restored, str(adapter_path)).eval()
    restored_head = getattr(restored.base_model.model, head_name)
    torch.testing.assert_close(restored_head.weight, head.weight, rtol=0, atol=0)
    torch.testing.assert_close(restored(input_ids, attention_mask=mask), expected, rtol=0.02, atol=0.01)

    apply_lora(str(base_path), str(adapter_path), str(merged_path), True, param_dtype)
    merged = get_llm_for_sequence_regression(
        str(merged_path), "reward", param_dtype=param_dtype, attn_implementation="eager"
    ).eval()
    assert getattr(merged, head_name).out_features == 1
    torch.testing.assert_close(getattr(merged, head_name).weight, head.weight, rtol=0, atol=0)
    assert merged.config.mean == 0.25 and merged.config.std == 1.75
    torch.testing.assert_close(merged(input_ids, attention_mask=mask), expected, rtol=0.03, atol=0.01)


def test_causal_lm_lora_merge_control(ds_module, tiny_lm, tmp_path):
    import torch
    from transformers import AutoModelForCausalLM

    from openrlhf.cli.lora_combiner import apply_lora

    base_path, tokenizer = tiny_lm
    model = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.bfloat16)
    model = peft.get_peft_model(model, peft.LoraConfig(r=2, target_modules=["q_proj", "v_proj"]))
    # Nonzero adapters ensure the test checks actual merging.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.fill_(0.1)
    model.eval()
    input_ids = torch.tensor([[2, 3, 4]])
    expected = model(input_ids).logits.detach()
    adapter_path, merged_path = tmp_path / "adapter", tmp_path / "merged"
    _strategy(ds_module, zero_stage=2).save_model(model, tokenizer, str(adapter_path))
    apply_lora(str(base_path), str(adapter_path), str(merged_path), False, "bf16")
    merged = AutoModelForCausalLM.from_pretrained(merged_path, torch_dtype=torch.bfloat16).eval()
    torch.testing.assert_close(merged(input_ids).logits, expected, rtol=0.03, atol=0.003)


def test_reward_lora_combiner_loads_scalar_head(ds_module, tiny_lm, tmp_path):
    import torch

    from openrlhf.cli.lora_combiner import apply_lora
    from openrlhf.models import get_llm_for_sequence_regression

    base_path, tokenizer = tiny_lm
    model = get_llm_for_sequence_regression(str(base_path), "reward", attn_implementation="eager")
    # Build a complete adapter independently of the reward-training LoRA configuration.
    model = peft.get_peft_model(
        model, peft.LoraConfig(r=2, target_modules=["q_proj", "v_proj"], modules_to_save=["score"])
    ).eval()
    adapter_path, merged_path = tmp_path / "adapter", tmp_path / "merged"
    _strategy(ds_module, zero_stage=2).save_model(model, tokenizer, str(adapter_path))
    apply_lora(str(base_path), str(adapter_path), str(merged_path), True, "bf16")
    merged = get_llm_for_sequence_regression(str(merged_path), "reward", attn_implementation="eager").eval()
    torch.testing.assert_close(merged.score.weight, model.base_model.model.score.weight, rtol=0, atol=0)


def test_reward_without_lora_and_critic_lora_control(ds_module, tiny_lm, tmp_path):
    import torch

    from openrlhf.models import get_llm_for_sequence_regression

    base_path, tokenizer = tiny_lm
    model = get_llm_for_sequence_regression(str(base_path), "reward", attn_implementation="eager")
    assert isinstance(model.score, nn.Linear) and model.score.weight.requires_grad
    output_path = tmp_path / "full"
    _strategy(ds_module, zero_stage=2).save_model(model, tokenizer, str(output_path))
    restored = get_llm_for_sequence_regression(str(output_path), "reward", attn_implementation="eager")
    torch.testing.assert_close(restored.score.weight, model.score.weight, rtol=0, atol=0)
    critic = get_llm_for_sequence_regression(
        str(base_path), "critic", lora_rank=2, target_modules=["q_proj", "v_proj"], attn_implementation="eager"
    )
    assert critic.peft_config["default"].modules_to_save is None
    assert isinstance(critic.base_model.model.score, nn.Linear)


def test_reward_merge_without_training_dependencies(tiny_lm, tmp_path):
    import subprocess
    from pathlib import Path

    base_path, _ = tiny_lm
    script = r"""
import importlib.abc
import sys
from pathlib import Path

class NoTrainingImports(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"flash_attn", "ring_flash_attn", "deepspeed"}:
            return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        raise AssertionError(f"Unneeded training import: {module.__name__}")

# Report packages as absent to optional-dependency probes, and fail any actual import.
find_spec = importlib.util.find_spec
importlib.util.find_spec = lambda name, *args, **kwargs: (
    None if name.split(".")[0] in {"flash_attn", "ring_flash_attn", "deepspeed"}
    else find_spec(name, *args, **kwargs)
)
sys.meta_path.insert(0, NoTrainingImports())
import torch
from transformers import AutoTokenizer
from openrlhf.cli.lora_combiner import apply_lora
from openrlhf.models import get_llm_for_sequence_regression

base, output = sys.argv[1:]
model = get_llm_for_sequence_regression(
    base, "reward", lora_rank=2, target_modules="all-linear", attn_implementation="eager"
).eval()
with torch.no_grad():
    model.base_model.model.score.weight.fill_(0.1)
ids = torch.tensor([[2, 3, 4]])
expected = model(ids, attention_mask=torch.ones_like(ids)).detach()
adapter, merged_path = Path(output) / "adapter", Path(output) / "merged"
model.save_pretrained(adapter)
model.config.to_json_file(adapter / "config.json")
AutoTokenizer.from_pretrained(base).save_pretrained(adapter)
apply_lora(base, str(adapter), str(merged_path), True, "bf16")
merged = get_llm_for_sequence_regression(str(merged_path), "reward", attn_implementation="eager").eval()
torch.testing.assert_close(merged.score.weight, model.base_model.model.score.weight, atol=0, rtol=0)
torch.testing.assert_close(merged(ids, attention_mask=torch.ones_like(ids)), expected, atol=0.01, rtol=0.03)
assert not any(name.split(".")[0] in {"flash_attn", "ring_flash_attn", "deepspeed"} for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(base_path), str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
