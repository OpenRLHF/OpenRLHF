import importlib
import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from transformers import (
    Lfm2Config,
    LlamaConfig,
    Mamba2Config,
    Qwen3_5Config,
    Qwen3_5MoeConfig,
    Qwen3Config,
    Qwen3NextConfig,
)

_TEST_PACKAGE = "_openrlhf_packing_compatibility_test"


def _load_model_modules():
    root = Path(__file__).resolve().parents[1]
    models_dir = root / "openrlhf" / "models"

    # Keep these constructor tests importable without the optional GPU stack.
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

    package = types.ModuleType(_TEST_PACKAGE)
    package.__path__ = [str(models_dir)]
    sys.modules[_TEST_PACKAGE] = package

    original_compile = torch.compile
    torch.compile = lambda fn=None, **_: fn if fn is not None else (lambda f: f)
    try:
        modules = {}
        for name in ("utils", "ring_attn_utils", "actor", "model"):
            full_name = f"{_TEST_PACKAGE}.{name}"
            spec = importlib.util.spec_from_file_location(full_name, models_dir / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[full_name] = module
            spec.loader.exec_module(module)
            modules[name] = module
    finally:
        torch.compile = original_compile
        for name in inserted_stubs:
            del sys.modules[name]

    return SimpleNamespace(**modules)


MODULES = _load_model_modules()
Actor = MODULES.actor.Actor


def _hybrid_configs():
    return [
        pytest.param(Qwen3_5Config(), id="qwen35-dense-text-config"),
        pytest.param(Qwen3_5MoeConfig(), id="qwen35-moe-text-config"),
        pytest.param(Qwen3NextConfig(), id="qwen3next-root-config"),
    ]


def _full_attention_config():
    return Qwen3Config(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        layer_types=["full_attention", "full_attention"],
    )


def _lfm2_config():
    return Lfm2Config(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        full_attn_idxs=[1],
    )


def _lfm2_attention_only_config():
    return Lfm2Config(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        full_attn_idxs=[0, 1],
    )


def _mamba2_config():
    return Mamba2Config(hidden_size=128, num_heads=2, head_dim=64, expand=1, num_hidden_layers=2)


def _mamba2_legacy_config():
    return SimpleNamespace(model_type="mamba2")


def _assert_incompatible_packing_error(call):
    with pytest.raises(ValueError) as raised:
        call()

    message = str(raised.value).lower()
    assert "cross-sample" in message or "across samples" in message
    assert "state" in message and ("leak" in message or "leakage" in message)
    assert "ds.packing_samples" in message
    assert "train.dynamic_batch_enable" in message
    assert "ring_attn_size" in message and "1" in message


def _patch_actor_loading(monkeypatch, config, *, loaded=None):
    monkeypatch.setattr(MODULES.actor, "set_z3_leaf_modules", lambda *_, **__: None)
    utils_module = importlib.import_module("openrlhf.utils.utils")
    monkeypatch.setattr(utils_module, "is_vlm_model", lambda *_: False)
    monkeypatch.setattr(
        MODULES.actor,
        "AutoConfig",
        SimpleNamespace(from_pretrained=lambda *_, **__: config),
        raising=False,
    )

    def load(*_, **__):
        if loaded is None:
            raise AssertionError("Actor attempted to load weights before validating packing compatibility")
        return loaded

    fake_auto_model = SimpleNamespace(from_pretrained=load)
    monkeypatch.setattr(MODULES.actor, "AutoModelForCausalLM", fake_auto_model)
    monkeypatch.setattr(MODULES.actor, "AutoModelForImageTextToText", fake_auto_model)


@pytest.mark.parametrize("config", _hybrid_configs())
def test_actor_pretrained_hybrid_packing_fails_before_loading(monkeypatch, config):
    _patch_actor_loading(monkeypatch, config)

    _assert_incompatible_packing_error(lambda: Actor("local-test-model", packing_samples=True))


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(_lfm2_config(), id="lfm2-convolution"),
        pytest.param(_mamba2_config(), id="mamba2-config"),
        pytest.param(_mamba2_legacy_config(), id="mamba2-legacy-no-layer-types"),
    ],
)
def test_actor_pretrained_stateful_packing_fails_before_loading(monkeypatch, config):
    _patch_actor_loading(monkeypatch, config)

    _assert_incompatible_packing_error(lambda: Actor("local-test-model", packing_samples=True))


@pytest.mark.parametrize(
    "config,packing_samples",
    [
        pytest.param(Qwen3_5Config(), False, id="hybrid-unpacked"),
        pytest.param(_full_attention_config(), True, id="full-attention-packed"),
        pytest.param(_lfm2_attention_only_config(), True, id="lfm2-attention-only-packed"),
        pytest.param(_lfm2_config(), False, id="lfm2-convolution-unpacked"),
        pytest.param(_mamba2_config(), False, id="mamba2-unpacked"),
        pytest.param(
            LlamaConfig(num_hidden_layers=2, hidden_size=128, intermediate_size=256),
            True,
            id="default-no-layer-types",
        ),
    ],
)
def test_actor_pretrained_allows_unpacked_or_attention_only(monkeypatch, config, packing_samples):
    loaded = torch.nn.Module()
    loaded.config = config
    _patch_actor_loading(monkeypatch, config, loaded=loaded)

    actor = Actor("local-test-model", packing_samples=packing_samples)
    assert actor.model is loaded


@pytest.mark.parametrize("config", _hybrid_configs())
def test_actor_module_hybrid_packing_fails(config):
    model = torch.nn.Module()
    model.config = config
    _assert_incompatible_packing_error(lambda: Actor(model, packing_samples=True))


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(_lfm2_config(), id="lfm2-convolution"),
        pytest.param(_mamba2_config(), id="mamba2-config"),
        pytest.param(_mamba2_legacy_config(), id="mamba2-legacy-no-layer-types"),
    ],
)
def test_actor_module_stateful_packing_fails(config):
    model = torch.nn.Module()
    model.config = config
    _assert_incompatible_packing_error(lambda: Actor(model, packing_samples=True))


@pytest.mark.parametrize(
    "config,packing_samples",
    [
        pytest.param(Qwen3_5Config(), False, id="hybrid-unpacked"),
        pytest.param(_full_attention_config(), True, id="full-attention-packed"),
        pytest.param(_lfm2_attention_only_config(), True, id="lfm2-attention-only-packed"),
        pytest.param(_lfm2_config(), False, id="lfm2-convolution-unpacked"),
        pytest.param(_mamba2_config(), False, id="mamba2-unpacked"),
        pytest.param(
            LlamaConfig(num_hidden_layers=2, hidden_size=128, intermediate_size=256),
            True,
            id="default-no-layer-types",
        ),
    ],
)
def test_actor_module_allows_unpacked_or_attention_only(config, packing_samples):
    model = torch.nn.Module()
    model.config = config
    actor = Actor(model, packing_samples=packing_samples)
    assert actor.model is not None


class _FakePretrainedModel(torch.nn.Module):
    base_model_prefix = "model"

    @classmethod
    def from_pretrained(cls, *_, **__):
        raise AssertionError("sequence-regression factory attempted to load weights before validation")


class _FakeBaseModel(_FakePretrainedModel):
    pass


def _patch_sequence_regression_factory(monkeypatch, config):
    monkeypatch.setattr(
        MODULES.model,
        "AutoConfig",
        SimpleNamespace(from_pretrained=lambda *_, **__: config),
    )
    monkeypatch.setattr(
        MODULES.model,
        "AutoModel",
        SimpleNamespace(_model_mapping={type(config): _FakeBaseModel}),
    )
    monkeypatch.setattr(MODULES.model, "set_z3_leaf_modules", lambda *_, **__: None)


@pytest.mark.parametrize("model_type", ["reward", "critic"])
@pytest.mark.parametrize(
    "config",
    [pytest.param(Qwen3_5Config(), id="nested-text-config"), pytest.param(Qwen3NextConfig(), id="root-config")],
)
def test_sequence_regression_factory_rejects_hybrid_packing_before_loading(monkeypatch, config, model_type):
    _patch_sequence_regression_factory(monkeypatch, config)

    _assert_incompatible_packing_error(
        lambda: MODULES.model.get_llm_for_sequence_regression("local-test-model", model_type, packing_samples=True)
    )


@pytest.mark.parametrize("model_type", ["reward", "critic"])
@pytest.mark.parametrize(
    "config",
    [
        pytest.param(_lfm2_config(), id="lfm2-convolution"),
        pytest.param(_mamba2_config(), id="mamba2-config"),
        pytest.param(_mamba2_legacy_config(), id="mamba2-legacy-no-layer-types"),
    ],
)
def test_sequence_regression_factory_rejects_stateful_packing_before_loading(monkeypatch, config, model_type):
    _patch_sequence_regression_factory(monkeypatch, config)

    _assert_incompatible_packing_error(
        lambda: MODULES.model.get_llm_for_sequence_regression("local-test-model", model_type, packing_samples=True)
    )


@pytest.mark.parametrize("model_type", ["reward", "critic"])
def test_sequence_regression_factory_allows_unpacked_hybrid(monkeypatch, model_type):
    config = Qwen3NextConfig()
    _patch_sequence_regression_factory(monkeypatch, config)

    loaded = torch.nn.Module()
    loaded.config = config
    monkeypatch.setattr(
        _FakePretrainedModel,
        "from_pretrained",
        classmethod(lambda cls, *_, **__: loaded),
    )

    assert (
        MODULES.model.get_llm_for_sequence_regression("local-test-model", model_type, packing_samples=False) is loaded
    )


@pytest.mark.parametrize("model_type", ["reward", "critic"])
@pytest.mark.parametrize(
    "config",
    [
        pytest.param(_lfm2_config(), id="lfm2-convolution"),
        pytest.param(_mamba2_config(), id="mamba2-config"),
    ],
)
def test_sequence_regression_factory_allows_unpacked_stateful(monkeypatch, config, model_type):
    _patch_sequence_regression_factory(monkeypatch, config)

    loaded = torch.nn.Module()
    loaded.config = config
    monkeypatch.setattr(
        _FakePretrainedModel,
        "from_pretrained",
        classmethod(lambda cls, *_, **__: loaded),
    )

    assert (
        MODULES.model.get_llm_for_sequence_regression("local-test-model", model_type, packing_samples=False) is loaded
    )


@pytest.mark.parametrize("model_type", ["reward", "critic"])
def test_sequence_regression_factory_allows_packed_full_attention(monkeypatch, model_type):
    config = _full_attention_config()
    _patch_sequence_regression_factory(monkeypatch, config)

    loaded = torch.nn.Module()
    loaded.config = config
    monkeypatch.setattr(
        _FakePretrainedModel,
        "from_pretrained",
        classmethod(lambda cls, *_, **__: loaded),
    )

    assert (
        MODULES.model.get_llm_for_sequence_regression("local-test-model", model_type, packing_samples=True) is loaded
    )
