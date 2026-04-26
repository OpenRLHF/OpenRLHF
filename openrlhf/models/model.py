from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, BitsAndBytesConfig

from openrlhf.utils.logging_utils import init_logger

from .actor import _build_peft_config_dict
from .ring_attn_utils import gather_and_pad_tensor, unpad_and_slice_tensor

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
        # Match dtype of the base for FSDP2 mp_policy compatibility.
        head = head.to(dtype=next(base.parameters()).dtype)
        setattr(self, value_head_prefix, head)

        self.register_buffer("mean", torch.zeros(1), persistent=False)
        self.register_buffer("std", torch.ones(1), persistent=False)
        if hasattr(config, "mean"):
            self.mean[0] = config.mean
            self.std[0] = config.std

    @property
    def base_transformer(self) -> nn.Module:
        # `LlamaForCausalLM.model` is the underlying `LlamaModel`; analogous for Qwen / Gemma3 / etc.
        # FSDP2 wraps modules in-place; attribute access still resolves.
        return getattr(self.base, self.base.base_model_prefix)


class RewardModel(_ValueHeadBase):
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_output: bool = False,
        ring_attn_group=None,
        pad_sequence: bool = False,
        packed_seq_lens=None,
    ) -> torch.Tensor:
        batch, seqlen = input_ids.size()
        eos_indices = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim=1, keepdim=True)
        forward_attention_mask = attention_mask
        if self.packing_samples:
            input_ids, position_ids, _, ring_attn_pad_len, indices = unpad_and_slice_tensor(
                input_ids, attention_mask, ring_attn_group
            )
            forward_attention_mask = None
        else:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)

        outputs = self.base_transformer(input_ids, attention_mask=forward_attention_mask, position_ids=position_ids)
        last_hidden_states = outputs["last_hidden_state"]
        values = getattr(self, self.value_head_prefix)(last_hidden_states).squeeze(-1)

        if self.packing_samples:
            values = gather_and_pad_tensor(values, ring_attn_group, ring_attn_pad_len, indices, batch, seqlen)
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
        ring_attn_group=None,
        values_allgather: bool = False,
        packed_seq_lens=None,
    ) -> torch.Tensor:
        batch, seqlen = input_ids.size()
        forward_attention_mask = attention_mask
        if self.packing_samples:
            input_ids, position_ids, _, ring_attn_pad_len, indices = unpad_and_slice_tensor(
                input_ids, attention_mask, ring_attn_group
            )
            forward_attention_mask = None
        else:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)

        outputs = self.base_transformer(input_ids, attention_mask=forward_attention_mask, position_ids=position_ids)

        if action_mask is None:
            assert return_output
            return outputs

        last_hidden_states = outputs["last_hidden_state"]
        values = getattr(self, self.value_head_prefix)(last_hidden_states).squeeze(-1)

        if self.packing_samples:
            values = gather_and_pad_tensor(values, ring_attn_group, ring_attn_pad_len, indices, batch, seqlen)

        values = values[:, :-1]
        if self.normalize_reward:
            values = (values - self.mean) / self.std

        action_values = values[:, -action_mask.shape[1] :] * action_mask.float()
        return (action_values, outputs) if return_output else action_values
