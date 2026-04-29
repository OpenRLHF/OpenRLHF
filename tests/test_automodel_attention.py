import pytest
import torch
from torch import nn

from openrlhf.models.actor import _resolve_custom_backend_attn, _validate_attn_implementation
from openrlhf.models.model import CriticModel, get_llm_for_sequence_regression
from openrlhf.utils.fsdp.packing import pack_padded_batch


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
            "attention_mask": attention_mask.detach().clone(),
            "position_ids": position_ids.detach().clone(),
            **kwargs,
        }
        hidden = input_ids.float().unsqueeze(-1)
        return type("Output", (), {"hidden_states": (hidden,)})()


def test_sequence_regression_hf_flash_packing_unpacks_values():
    model = _FakeSequenceRegressionModel()
    critic = CriticModel(model, "score", normalize_reward=False, packing_samples=True)
    input_ids = torch.tensor([[1, 2, 0], [3, 4, 5]])
    attention_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

    values, _ = critic._token_values(input_ids, attention_mask)

    torch.testing.assert_close(values, torch.tensor([[1.0, 2.0, 0.0], [3.0, 4.0, 5.0]]))
    torch.testing.assert_close(model.forward_kwargs["input_ids"], torch.tensor([[1, 2, 3, 4, 5]]))
    torch.testing.assert_close(model.forward_kwargs["position_ids"], torch.tensor([[0, 1, 0, 1, 2]]))
    torch.testing.assert_close(model.forward_kwargs["cu_seq_lens_q"], torch.tensor([0, 2, 5], dtype=torch.int32))


def test_sequence_regression_packing_requires_flash_attention_2():
    with pytest.raises(NotImplementedError, match="flash_attention_2"):
        get_llm_for_sequence_regression(
            "dummy-model",
            "critic",
            packing_samples=True,
            attn_implementation="sdpa",
        )
