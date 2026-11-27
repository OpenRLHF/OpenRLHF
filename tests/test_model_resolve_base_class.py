"""Tests for _resolve_base_class fallback for trust_remote_code configs.

Regression test for issue #1043: loading a reward/critic model whose config
class (e.g. Qwen2RMConfig, internlm/POLAR-7B) is not in
AutoModel._model_mapping raised KeyError in get_llm_for_sequence_regression,
while AutoModel.from_pretrained itself would resolve the same model through
the config's auto_map.
"""

from unittest.mock import patch

try:  # pragma: no cover - exercised implicitly by the import below
    import flash_attn  # noqa: F401
except ImportError:
    # ring_attn_utils imports flash_attn at module level; the class-resolution
    # logic under test does not touch it, so a minimal stub keeps this test
    # runnable on CPU-only environments. On CI (real flash_attn) the stub is
    # never installed.
    import sys
    import types
    from importlib.machinery import ModuleSpec

    def _mod(name):
        m = types.ModuleType(name)
        m.__spec__ = ModuleSpec(name, None)
        return m

    flash_attn = _mod("flash_attn")
    bert_padding = _mod("flash_attn.bert_padding")
    for attr in ("index_first_axis", "pad_input", "rearrange", "unpad_input"):
        setattr(bert_padding, attr, lambda *a, **k: None)
    distributed = _mod("flash_attn.utils.distributed")
    distributed.all_gather = lambda *a, **k: None
    utils_pkg = _mod("flash_attn.utils")
    utils_pkg.distributed = distributed
    flash_attn.bert_padding = bert_padding
    flash_attn.utils = utils_pkg
    sys.modules.update(
        {
            "flash_attn": flash_attn,
            "flash_attn.bert_padding": bert_padding,
            "flash_attn.utils": utils_pkg,
            "flash_attn.utils.distributed": distributed,
        }
    )

from transformers import AutoModel, Qwen2Config
from transformers.configuration_utils import PretrainedConfig

from openrlhf.models.model import _resolve_base_class


class Qwen2RMConfig(PretrainedConfig):
    """Mimics a custom trust_remote_code reward-model config."""

    model_type = "qwen2_rm"


SENTINEL = object()


def test_standard_config_uses_model_mapping():
    config = Qwen2Config()
    assert _resolve_base_class(config, "Qwen/Qwen2-0.5B") is AutoModel._model_mapping[Qwen2Config]


def test_custom_config_falls_back_to_auto_map():
    config = Qwen2RMConfig()
    config.auto_map = {"AutoModel": "some_remote_module.SomeModel"}

    with patch(
        "openrlhf.models.model.get_class_from_dynamic_module",
        return_value=SENTINEL,
    ) as mock_resolve:
        resolved = _resolve_base_class(config, "org/custom-reward-model")

    assert resolved is SENTINEL
    mock_resolve.assert_called_once_with("some_remote_module.SomeModel", "org/custom-reward-model")


def test_custom_config_falls_back_to_automodelforcausal_lm():
    config = Qwen2RMConfig()
    config.auto_map = {"AutoModelForCausalLM": "mod.Cls"}

    with patch(
        "openrlhf.models.model.get_class_from_dynamic_module",
        return_value=SENTINEL,
    ) as mock_resolve:
        resolved = _resolve_base_class(config, "org/custom-reward-model")

    assert resolved is SENTINEL
    mock_resolve.assert_called_once_with("mod.Cls", "org/custom-reward-model")


def test_custom_config_without_auto_map_still_raises():
    config = Qwen2RMConfig()
    try:
        _resolve_base_class(config, "org/custom-reward-model")
        raised = False
    except KeyError:
        raised = True
    assert raised
