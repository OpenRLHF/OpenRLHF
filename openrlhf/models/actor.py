from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM

from .ring_attn_utils import gather_and_pad_tensor, unpad_and_slice_tensor
from .utils import compute_entropy, log_probs_from_logits


class Actor(nn.Module):
    """
    Actor model for reinforcement learning.

    Builds an HF causal-LM on the ``meta`` device (no weight allocation) and
    optionally applies LoRA adapters.  Weights are loaded later via
    ``strategy.load_pretrained`` or ``load_hf_weights``.

    When ``use_meta_init=False``, the model is loaded with real weights via
    ``from_pretrained`` instead.  Combined with ``lora_path``, this allows
    loading a trained LoRA adapter and merging it into the base model for
    inference without a separate merge step.

    Args:
        pretrain (str): HuggingFace model name or local path.
        attn_implementation (str): Attention implementation. Defaults to ``"flash_attention_2"``.
        torch_dtype (torch.dtype): Model dtype for meta-init.
            Training models typically pass ``torch.float32`` (FSDP mixed-precision handles cast);
            ref/inference models pass ``torch.bfloat16`` or ``torch.float16``.
        use_meta_init (bool): If ``True`` (default), build model on meta device
            (weights loaded later by strategy). If ``False``, load real weights
            via ``from_pretrained``.
        use_liger_kernel (bool): Use Liger Kernel optimised causal LM. Defaults to ``False``.
        lora_rank (int): LoRA rank. 0 disables LoRA. Defaults to 0.
        lora_alpha (int): LoRA alpha. Defaults to 16.
        lora_dropout (float): LoRA dropout. Defaults to 0.
        target_modules (list | None): LoRA target modules. Defaults to ``None``.
        lora_path (str | None): Path to a trained LoRA adapter directory.
            When set (requires ``use_meta_init=False``), the adapter is loaded
            via ``PeftModel.from_pretrained`` and merged into the base model.
        packing_samples (bool): Pack samples during training. Defaults to ``False``.
        temperature (float): Temperature for log-prob computation. Defaults to 1.0.
    """

    def __init__(
        self,
        pretrain: str,
        *,
        attn_implementation="flash_attention_2",
        torch_dtype: torch.dtype = torch.float32,
        use_meta_init: bool = True,
        lora_rank=0,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=None,
        lora_path=None,
        packing_samples=False,
        temperature=1.0,
        use_liger_kernel=False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.packing_samples = packing_samples

        if use_meta_init:
            # -- Build causal-LM structure on meta device (no weight allocation) --
            cfg = AutoConfig.from_pretrained(pretrain, trust_remote_code=True)
            cfg.use_cache = False
            try:
                cfg._attn_implementation = attn_implementation
            except Exception:
                setattr(cfg, "_attn_implementation", attn_implementation)

            with torch.device("meta"):
                if use_liger_kernel:
                    from liger_kernel.transformers import AutoLigerKernelForCausalLM

                    model = AutoLigerKernelForCausalLM.from_config(
                        cfg, dtype=torch_dtype, trust_remote_code=True
                    )
                else:
                    model = AutoModelForCausalLM.from_config(
                        cfg, dtype=torch_dtype, trust_remote_code=True
                    )

            self.model = model
            if hasattr(self.model, "config") and hasattr(self.model.config, "use_cache"):
                self.model.config.use_cache = False

            # LoRA (meta-init compatible): apply adapters before TP/FSDP wrapping.
            if lora_rank > 0:
                self.model.enable_input_require_grads()
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    target_modules=target_modules,
                    lora_dropout=lora_dropout,
                    bias="none",
                )
                self.model = get_peft_model(self.model, lora_config)
        else:
            # -- Non meta-init: load real weights via from_pretrained --
            from peft import PeftModel

            model = AutoModelForCausalLM.from_pretrained(
                pretrain,
                dtype=torch_dtype,
                attn_implementation=attn_implementation,
                trust_remote_code=True,
            )
            if lora_path:
                model = PeftModel.from_pretrained(model, lora_path, dtype=torch_dtype)
                model = model.merge_and_unload()

            self.model = model

        # MoE - balancing loss
        model_config = self.model.config.to_dict()
        if "output_router_logits" in model_config:
            print("[MoE] set output_router_logits as True")
            self.model.config.output_router_logits = True

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
    ) -> torch.Tensor:
        """Returns action log probs"""
        batch, seqlen = sequences.size()
        foward_attention_mask = attention_mask
        if self.packing_samples:
            sequences, position_ids, rolled_sequences, ring_attn_pad_len, indices = unpad_and_slice_tensor(
                sequences, attention_mask, ring_attn_group
            )
            foward_attention_mask = None
        else:
            # https://github.com/OpenRLHF/OpenRLHF/issues/217
            rolled_sequences = torch.roll(sequences, shifts=-1, dims=1)
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)

        output = self.model(sequences, attention_mask=foward_attention_mask, position_ids=position_ids)
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

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs={"use_reentrant": False}):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()

    def print_trainable_parameters(self):
        self.model.print_trainable_parameters()
