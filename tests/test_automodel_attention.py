import sys
import types

import pytest
import torch
from torch import nn

from openrlhf.models.actor import (
    _automodel_custom_supports_thd_packing,
    _build_peft_config_dict,
    _resolve_custom_backend_attn,
    _validate_attn_implementation,
)
from openrlhf.models.model import CriticModel, RewardModel, get_llm_for_sequence_regression
from openrlhf.utils.fsdp.packing import _restore_cp_chunks, pack_padded_batch, pad_to_cp_multiple


def test_custom_backend_maps_hf_attention_to_sdpa():
    assert _resolve_custom_backend_attn("flash_attention_2", packing_samples=False) == "sdpa"
    assert _resolve_custom_backend_attn("eager", packing_samples=False) == "sdpa"


def test_custom_thd_packing_accepts_te_attention():
    assert _resolve_custom_backend_attn("te", packing_samples=True) == "te"


@pytest.mark.parametrize("attn_implementation", ["sdpa", "flex", "flash_attention_2"])
def test_custom_thd_packing_requires_te_attention(attn_implementation):
    with pytest.raises(ValueError, match="packing_samples"):
        _resolve_custom_backend_attn(attn_implementation, packing_samples=True)


def test_validate_rejects_unknown_attention_backend():
    with pytest.raises(ValueError, match="Unsupported attention implementation"):
        _validate_attn_implementation("not_an_attention_backend")


def test_lora_all_linear_list_sentinel_matches_all_linear():
    cfg = _build_peft_config_dict(4, 8, 0.1, ["all-linear"])

    assert cfg.match_all_linear is True
    assert cfg.target_modules == []


def test_lora_explicit_target_modules_are_preserved():
    cfg = _build_peft_config_dict(4, 8, 0.1, ["q_proj", "v_proj"])

    assert cfg.match_all_linear is False
    assert cfg.target_modules == ["q_proj", "v_proj"]


def test_pack_padded_batch_hf_flash_attention_kwargs():
    input_ids = torch.tensor([[1, 2, 0], [3, 4, 5]])
    attention_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

    packed_ids, position_ids, rolled_ids, indices, attn_kwargs = pack_padded_batch(
        input_ids, attention_mask, style="hf"
    )

    torch.testing.assert_close(packed_ids, torch.tensor([[1, 2, 3, 4, 5]]))
    torch.testing.assert_close(position_ids, torch.tensor([[0, 1, 0, 1, 2]]))
    torch.testing.assert_close(rolled_ids, torch.tensor([[2, 0, 4, 5, 3]]))
    torch.testing.assert_close(indices, torch.tensor([0, 1, 3, 4, 5]))
    assert set(attn_kwargs) == {"cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"}
    torch.testing.assert_close(attn_kwargs["cu_seq_lens_q"], torch.tensor([0, 2, 5], dtype=torch.int32))
    assert attn_kwargs["max_length_q"] == 3


def test_pack_padded_batch_automodel_thd_kwargs():
    input_ids = torch.tensor([[1, 2, 0], [3, 4, 5]])
    attention_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

    *_, attn_kwargs = pack_padded_batch(input_ids, attention_mask, style="automodel")

    assert attn_kwargs["qkv_format"] == "thd"
    torch.testing.assert_close(attn_kwargs["cu_seqlens"], torch.tensor([0, 2, 5], dtype=torch.int32))
    assert attn_kwargs["max_seqlen"] == 3


def test_restore_cp_chunks_undoes_automodel_load_balancing():
    # cp_size=2 rank chunks carry original chunks [0, 3] and [1, 2].
    rank0 = torch.tensor([[0, 3]])
    rank1 = torch.tensor([[1, 2]])

    restored = _restore_cp_chunks([rank0, rank1], cp_size=2, seq_dim=1)

    torch.testing.assert_close(restored, torch.tensor([[0, 1, 2, 3]]))


def test_pad_to_cp_multiple_right_pads_to_two_cp_chunks():
    tensor = torch.tensor([[1, 2, 3, 4, 5]])

    padded = pad_to_cp_multiple(tensor, cp_size=2, seq_dim=1, value=0)

    torch.testing.assert_close(padded, torch.tensor([[1, 2, 3, 4, 5, 0, 0, 0]]))


def test_vllm_refit_adapter_uses_to_hf_single_tensor_fallback():
    from openrlhf.trainer.ray.ppo_actor import _maybe_adapt_tensor_to_hf

    class Adapter:
        def to_hf(self, state_dict, **kwargs):
            assert kwargs["exclude_key_regex"] == r".*_extra_state.*"
            return {f"hf.{k}": v + 1 for k, v in state_dict.items()}

    class Model:
        state_dict_adapter = Adapter()

    tensor = torch.tensor([1.0])

    converted = _maybe_adapt_tensor_to_hf(Model(), "model.weight", tensor)

    assert len(converted) == 1
    name, value = converted[0]
    assert name == "hf.model.weight"
    torch.testing.assert_close(value, torch.tensor([2.0]))


class _FakeSequenceRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {})()
        self.score = nn.Linear(1, 1, bias=False)
        nn.init.ones_(self.score.weight)
        self.forward_kwargs = None

    def forward(self, input_ids, attention_mask=None, position_ids=None, **kwargs):
        self.forward_kwargs = {
            "input_ids": input_ids.detach().clone(),
            "attention_mask": None if attention_mask is None else attention_mask.detach().clone(),
            "position_ids": position_ids.detach().clone(),
            **kwargs,
        }
        hidden = input_ids.float().unsqueeze(-1)
        return type("Output", (), {"hidden_states": (hidden,)})()


class _FakeLogitSequenceRegressionModel(_FakeSequenceRegressionModel):
    def forward(self, input_ids, attention_mask=None, position_ids=None, **kwargs):
        self.forward_kwargs = {
            "input_ids": input_ids.detach().clone(),
            "attention_mask": None if attention_mask is None else attention_mask.detach().clone(),
            "position_ids": position_ids.detach().clone(),
            **kwargs,
        }
        logits = (input_ids.float() + 10).unsqueeze(-1)
        hidden = input_ids.float().unsqueeze(-1)
        return type("Output", (), {"logits": logits, "hidden_states": (hidden,)})()


class _FakeConfig:
    hidden_size = 1
    model_type = "fake"
    architectures = []

    def to_dict(self):
        return {}


class _FakeHFSequenceRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _FakeConfig()
        self.score = nn.Linear(1, 1, bias=False)

    def forward(self, *args, **kwargs):
        raise AssertionError("not used by construction tests")


class _FakeCustomNoHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _FakeConfig()


_FakeCustomNoHead.__module__ = "nemo_automodel.components.models.fake"


class _FakeCustomSequenceRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _FakeConfig()
        self.score = nn.Linear(1, 1, bias=False)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        output_hidden_states=False,
        use_cache=False,
        return_dict=True,
        **kwargs,
    ):
        hidden = input_ids.float().unsqueeze(-1)
        return type("Output", (), {"hidden_states": (hidden,)})()


_FakeCustomSequenceRegressionModel.__module__ = "nemo_automodel.components.models.fake"


class _FakeCustomThdModel(nn.Module):
    pass


_FakeCustomThdModel.__module__ = "nemo_automodel.components.models.fake"


def _install_fake_sequence_classification_automodel(monkeypatch, calls, first_model=None):
    class FakeNeMoAutoModelForSequenceClassification:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls.append(kwargs)
            if not kwargs["force_hf"] and first_model is not None:
                return first_model()
            return _FakeHFSequenceRegressionModel()

    fake_nemo = types.ModuleType("nemo_automodel")
    fake_nemo.NeMoAutoModelForSequenceClassification = FakeNeMoAutoModelForSequenceClassification
    monkeypatch.setitem(sys.modules, "nemo_automodel", fake_nemo)


def test_custom_model_thd_support_is_capability_checked(monkeypatch):
    from openrlhf.models import actor as actor_mod

    monkeypatch.setattr(
        actor_mod,
        "_class_source_supports_thd_packing",
        lambda cls: cls is _FakeCustomThdModel,
    )

    assert _automodel_custom_supports_thd_packing(_FakeCustomThdModel())
    assert not _automodel_custom_supports_thd_packing(_FakeCustomNoHead())


def test_sequence_regression_hf_flash_packing_unpacks_values():
    model = _FakeSequenceRegressionModel()
    critic = CriticModel(model, "score", normalize_reward=False, packing_samples=True)
    input_ids = torch.tensor([[1, 2, 0], [3, 4, 5]])
    attention_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

    values, _ = critic._token_values(input_ids, attention_mask)

    torch.testing.assert_close(values, torch.tensor([[1.0, 2.0, 0.0], [3.0, 4.0, 5.0]]))
    torch.testing.assert_close(model.forward_kwargs["input_ids"], torch.tensor([[1, 2, 3, 4, 5]]))
    assert model.forward_kwargs["attention_mask"] is None
    torch.testing.assert_close(model.forward_kwargs["position_ids"], torch.tensor([[0, 1, 0, 1, 2]]))
    torch.testing.assert_close(model.forward_kwargs["cu_seq_lens_q"], torch.tensor([0, 2, 5], dtype=torch.int32))


def test_sequence_regression_prefers_token_logits_when_available():
    model = _FakeLogitSequenceRegressionModel()
    critic = CriticModel(model, "score", normalize_reward=False)
    input_ids = torch.tensor([[1, 2, 0], [3, 4, 5]])
    attention_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

    values, _ = critic._token_values(input_ids, attention_mask)

    torch.testing.assert_close(values, torch.tensor([[11.0, 12.0, 10.0], [13.0, 14.0, 15.0]]))


def test_sequence_regression_custom_without_head_falls_back_to_hf(monkeypatch):
    from openrlhf.models import model as model_mod

    calls = []
    _install_fake_sequence_classification_automodel(monkeypatch, calls, first_model=_FakeCustomNoHead)
    monkeypatch.setattr(model_mod.AutoConfig, "from_pretrained", lambda *args, **kwargs: _FakeConfig())
    monkeypatch.setattr(model_mod, "_will_use_hf_model", lambda *args, **kwargs: False)
    monkeypatch.setattr(model_mod, "_custom_sequence_regression_backend_kwargs", lambda attn: ({}, "sdpa"))

    wrapper = get_llm_for_sequence_regression("dummy-model", "reward", attn_implementation="sdpa")

    assert isinstance(wrapper.model, _FakeHFSequenceRegressionModel)
    assert [call["force_hf"] for call in calls] == [False, True]
    assert [call["has_packed_sequence"] for call in calls] == [False, False]


@pytest.mark.parametrize("model_type,wrapper_cls", [("reward", RewardModel), ("critic", CriticModel)])
def test_sequence_regression_custom_path_is_used_for_reward_and_critic(monkeypatch, model_type, wrapper_cls):
    from openrlhf.models import model as model_mod

    calls = []
    _install_fake_sequence_classification_automodel(
        monkeypatch,
        calls,
        first_model=_FakeCustomSequenceRegressionModel,
    )
    monkeypatch.setattr(model_mod.AutoConfig, "from_pretrained", lambda *args, **kwargs: _FakeConfig())
    monkeypatch.setattr(model_mod, "_will_use_hf_model", lambda *args, **kwargs: False)
    monkeypatch.setattr(model_mod, "_custom_sequence_regression_backend_kwargs", lambda attn: ({}, "sdpa"))

    wrapper = get_llm_for_sequence_regression("dummy-model", model_type, attn_implementation="sdpa")

    assert isinstance(wrapper, wrapper_cls)
    assert isinstance(wrapper.model, _FakeCustomSequenceRegressionModel)
    assert len(calls) == 1
    assert calls[0]["force_hf"] is False
    assert calls[0]["has_packed_sequence"] is False
    assert wrapper._forward_autocast_dtype is torch.bfloat16


def test_sequence_regression_packing_skips_custom_and_uses_hf_flash(monkeypatch):
    from openrlhf.models import model as model_mod

    calls = []
    _install_fake_sequence_classification_automodel(monkeypatch, calls, first_model=_FakeCustomNoHead)
    monkeypatch.setattr(model_mod.AutoConfig, "from_pretrained", lambda *args, **kwargs: _FakeConfig())
    monkeypatch.setattr(model_mod, "_will_use_hf_model", lambda *args, **kwargs: False)
    monkeypatch.setattr(model_mod, "_has_hf_flash_attn_2", lambda: True)

    wrapper = get_llm_for_sequence_regression(
        "dummy-model",
        "reward",
        packing_samples=True,
        attn_implementation="te",
    )

    assert isinstance(wrapper.model, _FakeHFSequenceRegressionModel)
    assert len(calls) == 1
    assert calls[0]["force_hf"] is True
    assert calls[0]["has_packed_sequence"] is True
    assert calls[0]["attn_implementation"] == "flash_attention_2"


def test_sequence_regression_custom_architecture_is_used_for_llama_regression(monkeypatch):
    from openrlhf.models import model as model_mod

    class _FakeLlamaConfig(_FakeConfig):
        model_type = "llama"
        architectures = ["LlamaForCausalLM"]

    calls = []
    _install_fake_sequence_classification_automodel(
        monkeypatch,
        calls,
        first_model=_FakeCustomSequenceRegressionModel,
    )
    monkeypatch.setattr(model_mod.AutoConfig, "from_pretrained", lambda *args, **kwargs: _FakeLlamaConfig())
    monkeypatch.setattr(model_mod, "_will_use_hf_model", lambda *args, **kwargs: False)
    monkeypatch.setattr(model_mod, "_custom_sequence_regression_backend_kwargs", lambda attn: ({}, "sdpa"))
    monkeypatch.setattr(
        model_mod,
        "_register_openrlhf_llama_sequence_regression",
        lambda config: "OpenRLHFLlamaForCausalSequenceRegression",
    )

    wrapper = get_llm_for_sequence_regression("dummy-model", "critic", attn_implementation="sdpa")

    assert isinstance(wrapper.model, _FakeCustomSequenceRegressionModel)
    assert calls[0]["config"].architectures == ["OpenRLHFLlamaForCausalSequenceRegression"]


def test_sequence_regression_packing_requires_flash_attention_2():
    with pytest.raises(NotImplementedError, match="flash_attention_2"):
        get_llm_for_sequence_regression(
            "dummy-model",
            "critic",
            packing_samples=True,
            attn_implementation="sdpa",
        )
