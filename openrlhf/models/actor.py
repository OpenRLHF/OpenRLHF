from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import BitsAndBytesConfig

from .ring_attn_utils import gather_and_pad_tensor, unpad_and_slice_tensor
from .utils import compute_entropy, log_probs_from_logits


def _build_peft_config_dict(rank: int, alpha: int, dropout: float, target_modules):
    """Map OpenRLHF lora.* args onto Automodel's PeftConfig schema.

    Field-name gotcha: Automodel renames `r` (LoRA rank) → `dim`.
    """
    base = {"dim": rank, "alpha": alpha, "dropout": dropout}
    # Map HF-peft sentinel "all-linear" → Automodel's `match_all_linear=True`.
    if not target_modules or target_modules == "all-linear":
        return {**base, "match_all_linear": True}
    if isinstance(target_modules, str):
        target_modules = [target_modules]
    return {**base, "target_modules": list(target_modules)}


class Actor(nn.Module):
    """Actor wrapper for RLHF training.

    Builds the underlying model via Automodel's official entry
    (``NeMoAutoModelForCausalLM.from_pretrained`` / ``NeMoAutoModelForImageTextToText``),
    which in a single call: loads HF weights, applies the per-architecture TP plan,
    wraps with FSDP2 over ``device_mesh``, attaches CP hooks if cp_size>1, and
    optionally applies LoRA + quantization + activation checkpointing.
    """

    def __init__(
        self,
        pretrain_or_model,
        attn_implementation: str = "flash_attention_2",
        param_dtype: str = "bf16",
        load_in_4bit: bool = False,
        lora_rank: int = 0,
        lora_alpha: int = 16,
        lora_dropout: float = 0,
        target_modules=None,
        device_mesh=None,
        distributed_config=None,
        activation_checkpointing: bool = False,
        packing_samples: bool = False,
        temperature: float = 1.0,
        use_liger_kernel: bool = False,
        freeze_visual_encoder: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.packing_samples = packing_samples

        if not isinstance(pretrain_or_model, str):
            self.model = pretrain_or_model
            return

        from openrlhf.utils.utils import convert_to_torch_dtype, is_vlm_model

        torch_dtype = convert_to_torch_dtype(param_dtype)
        self.is_vlm = is_vlm_model(pretrain_or_model)

        if self.is_vlm and use_liger_kernel:
            raise ValueError(
                "use_liger_kernel is not compatible with VLM models. "
                "Liger kernel only supports CausalLM, not ImageTextToText."
            )

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

        if self.is_vlm:
            from nemo_automodel import NeMoAutoModelForImageTextToText as ModelCls
        else:
            from nemo_automodel import NeMoAutoModelForCausalLM as ModelCls

        self.model = ModelCls.from_pretrained(
            pretrain_or_model,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            quantization_config=nf4_config,
            device_mesh=device_mesh,
            distributed_config=distributed_config,
            activation_checkpointing=activation_checkpointing,
            peft_config=peft_config,
            use_liger_kernel=use_liger_kernel and not self.is_vlm,
            has_packed_sequence=packing_samples,
        )

        # VLM: optionally freeze the vision encoder so only the language
        # model backbone is trained. Both Qwen3.5 and Gemma4 place language
        # params under "language_model.*" / "lm_head.*"; everything else
        # (visual encoder, projector) gets frozen.
        if self.is_vlm and freeze_visual_encoder:
            for name, param in self.model.named_parameters():
                if "language_model" not in name and "lm_head" not in name:
                    param.requires_grad = False

        # MoE - balancing loss
        if "output_router_logits" in self.model.config.to_dict():
            print("[MoE] set output_router_logits as True")
            self.model.config.output_router_logits = True

        # https://github.com/huggingface/transformers/issues/26877
        # Use `model.generate(use_cache=True)` instead.
        self.model.config.use_cache = False

        if self.is_vlm:
            self._vlm_config = self.model.config

    def forward(
        self,
        sequences: torch.LongTensor,
        action_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_output=False,
        allgather_logits=False,
        return_logprobs=False,
        ring_attn_group: Optional[dist.ProcessGroup] = None,
        packed_seq_lens: Optional[list[int]] = None,
        return_entropy=False,
        **mm_inputs,
    ) -> torch.Tensor:
        """Returns action log probs."""
        batch, seqlen = sequences.size()
        if self.packing_samples:
            sequences, position_ids, rolled_sequences, ring_attn_pad_len, indices = unpad_and_slice_tensor(
                sequences, attention_mask, ring_attn_group
            )
            foward_attention_mask = None
        else:
            # https://github.com/OpenRLHF/OpenRLHF/issues/217
            rolled_sequences = torch.roll(sequences, shifts=-1, dims=1)
            foward_attention_mask = attention_mask

            if getattr(self, "is_vlm", False):
                position_ids = None
                if mm_inputs:
                    cfg = self._vlm_config
                    token_type_ids = (sequences == cfg.image_token_id).to(torch.int32)
                    if getattr(cfg, "video_token_id", None) is not None:
                        token_type_ids[sequences == cfg.video_token_id] = 2
                    key = "mm_token_type_ids" if "image_grid_thw" in mm_inputs else "token_type_ids"
                    mm_inputs[key] = token_type_ids
            else:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)

        output = self.model(sequences, attention_mask=foward_attention_mask, position_ids=position_ids, **mm_inputs)
        # https://github.com/OpenRLHF/OpenRLHF/pull/634
        output["logits"] = output["logits"].to(torch.float32)

        if return_entropy:
            assert return_output
            entropy = compute_entropy(output["logits"])
            if self.packing_samples:
                entropy = gather_and_pad_tensor(entropy, ring_attn_group, ring_attn_pad_len, indices, batch, seqlen)
            setattr(output, "entropy", entropy[:, :-1])

        return_action_log_probs = action_mask is not None
        if not return_action_log_probs and not return_logprobs:
            assert return_output
            if allgather_logits and self.packing_samples:
                output["logits"] = gather_and_pad_tensor(
                    output["logits"], ring_attn_group, ring_attn_pad_len, indices, batch, seqlen
                )
            return output

        log_probs = log_probs_from_logits(output["logits"], rolled_sequences, temperature=self.temperature)

        if self.packing_samples:
            log_probs = gather_and_pad_tensor(log_probs, ring_attn_group, ring_attn_pad_len, indices, batch, seqlen)

        log_probs = log_probs[:, :-1]
        if not return_action_log_probs and return_logprobs:
            return (log_probs, output) if return_output else log_probs

        action_log_probs = log_probs[:, -action_mask.shape[1] :] * action_mask.float()
        return (action_log_probs, output) if return_output else action_log_probs

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs={"use_reentrant": False}):
        # No-op under FSDP/Automodel: activation checkpointing is configured
        # at construction time via `activation_checkpointing=True` on
        # `NeMoAutoModelForCausalLM.from_pretrained`. Calling HF's late hook
        # would conflict with FSDP2's already-applied wrap.
        pass

    def gradient_checkpointing_disable(self):
        pass

    def print_trainable_parameters(self):
        if hasattr(self.model, "print_trainable_parameters"):
            self.model.print_trainable_parameters()
