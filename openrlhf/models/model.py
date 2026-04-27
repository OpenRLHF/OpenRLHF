from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, BitsAndBytesConfig

from openrlhf.utils.fsdp.packing import pack_padded_batch, unpack_to_padded, unshard_dtensor
from openrlhf.utils.logging_utils import init_logger

from .actor import _build_peft_config_dict

logger = init_logger(__name__)


def get_llm_for_sequence_regression(
    model_name_or_path: str,
    model_type: str,
    *,
    param_dtype: str = "bf16",
    load_in_4bit: bool = False,
    lora_rank: int = 0,
    lora_alpha: int = 16,
    target_modules=None,
    lora_dropout: float = 0,
    normalize_reward: bool = False,
    attn_implementation: str = "flash_attention_2",
    device_mesh=None,
    distributed_config=None,
    activation_checkpointing: bool = False,
    init_value_head: bool = False,
    value_head_prefix: str = "score",
    packing_samples: bool = False,
    use_liger_kernel: bool = False,
    **kwargs,
) -> nn.Module:
    """Build a reward / critic model on top of an Automodel-parallelized base.

    Loads the base CausalLM via ``NeMoAutoModelForCausalLM.from_pretrained``
    (FSDP2 + TP/CP/EP applied per the device_mesh) and composes a per-token
    value head over it. The CausalLM's lm_head is loaded but not invoked at
    forward time — wasted memory for MVP; Phase 5 may swap in a base-only path.
    """
    assert model_type in ("critic", "reward"), f"invalid model_type: {model_type}"

    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    config.normalize_reward = normalize_reward
    value_head_prefix = getattr(config, "value_head_prefix", value_head_prefix)
    logger.info(f"set value_head_prefix to `{value_head_prefix}`")

    from openrlhf.utils.utils import convert_to_torch_dtype

    torch_dtype = convert_to_torch_dtype(param_dtype)

    nf4_config = None
    if load_in_4bit:
        assert param_dtype == "bf16", "we only support bnb_4bit_compute_dtype = bf16"
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    peft_config = None
    if lora_rank > 0:
        peft_config = _build_peft_config_dict(lora_rank, lora_alpha, lora_dropout, target_modules)

    from nemo_automodel import NeMoAutoModelForCausalLM

    base = NeMoAutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        quantization_config=nf4_config,
        device_mesh=device_mesh,
        distributed_config=distributed_config,
        activation_checkpointing=activation_checkpointing,
        peft_config=peft_config,
        use_liger_kernel=use_liger_kernel,
        has_packed_sequence=packing_samples,
    )

    if "output_router_logits" in base.config.to_dict():
        print("[MoE] set output_router_logits as True")
        base.config.output_router_logits = True
    base.config.use_cache = False

    cls = RewardModel if model_type == "reward" else CriticModel
    wrapped = cls(
        base=base,
        config=config,
        value_head_prefix=value_head_prefix,
        packing_samples=packing_samples,
        normalize_reward=normalize_reward,
        init_value_head=init_value_head,
    )
    return wrapped


class _ValueHeadBase(nn.Module):
    """Common scaffolding for RewardModel and CriticModel.

    Composes an Automodel-parallelized CausalLM with a per-token value head.
    The value head is plain (replicated across DP, not TP-sharded) — tiny
    relative to the base, so the cost of replication is negligible.
    """

    def __init__(
        self,
        base: nn.Module,
        config,
        value_head_prefix: str,
        packing_samples: bool,
        normalize_reward: bool,
        init_value_head: bool,
    ) -> None:
        super().__init__()
        self.base = base
        self.config = config
        self.value_head_prefix = value_head_prefix
        self.packing_samples = packing_samples
        self.normalize_reward = normalize_reward

        # Register the value head under the configured prefix (e.g., "score").
        head = nn.Linear(config.hidden_size, 1, bias=False)
        if init_value_head:
            head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))
        # Match dtype + device of the base for FSDP2 mp_policy compatibility.
        ref = next(base.parameters())
        head = head.to(device=ref.device, dtype=ref.dtype)
        setattr(self, value_head_prefix, head)

        self.register_buffer("mean", torch.zeros(1), persistent=False)
        self.register_buffer("std", torch.ones(1), persistent=False)
        if hasattr(config, "mean"):
            self.mean[0] = config.mean
            self.std[0] = config.std

    def _forward_base(self, input_ids, attention_mask, position_ids, **fa_kwargs):
        """Run the root CausalLM forward (so FSDP2's root unshard hook fires for
        embed_tokens / final norm) and return hidden states. We discard ``logits``
        (a wasted lm_head matmul) — Phase 5 may swap to a base-only path.
        """
        out = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            use_cache=False,
            **fa_kwargs,
        )
        # `out.last_hidden_state` is set when output_hidden_states=True; falls back
        # to the final hidden_states tuple element for older HF / custom models.
        last = getattr(out, "last_hidden_state", None)
        if last is None:
            last = out.hidden_states[-1]
        return last, out


class RewardModel(_ValueHeadBase):
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_output: bool = False,
        pad_sequence: bool = False,
        packed_seq_lens=None,
    ) -> torch.Tensor:
        batch, seqlen = input_ids.size()
        eos_indices = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim=1, keepdim=True)
        forward_attention_mask = attention_mask
        fa_kwargs: dict = {}
        indices = None
        if self.packing_samples:
            input_ids, position_ids, _, indices, fa_kwargs = pack_padded_batch(input_ids, attention_mask)
            forward_attention_mask = None
        else:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)

        last_hidden_states, outputs = self._forward_base(
            input_ids, attention_mask=forward_attention_mask, position_ids=position_ids, **fa_kwargs
        )
        # Under SP / TP, `last_hidden_state` may be a DTensor (sequence-sharded
        # under SP; replicated otherwise). The value head is a plain `nn.Linear`
        # not registered for TP and downstream `.gather(dim=1, eos_indices)`
        # doesn't support DTensor — unshard before applying.
        last_hidden_states = unshard_dtensor(last_hidden_states)
        values = getattr(self, self.value_head_prefix)(last_hidden_states).squeeze(-1)

        if self.packing_samples:
            values = unpack_to_padded(values, indices, batch, seqlen)
        reward = values.gather(dim=1, index=eos_indices).squeeze(1)

        if not self.training and self.normalize_reward:
            reward = (reward - self.mean) / self.std

        return (reward, outputs) if return_output else reward


class CriticModel(_ValueHeadBase):
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        action_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_output: bool = False,
        values_allgather: bool = False,
        packed_seq_lens=None,
    ) -> torch.Tensor:
        batch, seqlen = input_ids.size()
        forward_attention_mask = attention_mask
        fa_kwargs: dict = {}
        indices = None
        if self.packing_samples:
            input_ids, position_ids, _, indices, fa_kwargs = pack_padded_batch(input_ids, attention_mask)
            forward_attention_mask = None
        else:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)

        last_hidden_states, outputs = self._forward_base(
            input_ids, attention_mask=forward_attention_mask, position_ids=position_ids, **fa_kwargs
        )

        if action_mask is None:
            assert return_output
            return outputs

        # See RewardModel for SP unshard rationale.
        last_hidden_states = unshard_dtensor(last_hidden_states)
        values = getattr(self, self.value_head_prefix)(last_hidden_states).squeeze(-1)

        if self.packing_samples:
            values = unpack_to_padded(values, indices, batch, seqlen)

        values = values[:, :-1]
        if self.normalize_reward:
            values = (values - self.mean) / self.std

        action_values = values[:, -action_mask.shape[1] :] * action_mask.float()
        return (action_values, outputs) if return_output else action_values
