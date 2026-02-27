from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

from .ring_attn_utils import gather_and_pad_tensor, unpad_and_slice_tensor
from .utils import compute_entropy, compute_token_log_probs


class Actor(nn.Module):
    """
    Actor model for reinforcement learning.

    - ``device_map="meta"`` (default): build an HF causal-LM on the ``meta``
      device (no weight allocation). Weights are loaded later via
      ``strategy.load_pretrained``.
    - ``device_map!="meta"``: load real weights via HuggingFace
      ``from_pretrained`` and forward ``device_map`` to it (e.g. ``"auto"``,
      ``"cuda:0"``, or ``"cpu"``). If ``device_map is None``, HF defaults are
      used (typically CPU).

    Args:
        pretrain (str): HuggingFace model name or local path.
        attn_implementation (str): Attention implementation. Defaults to ``"flash_attention_2"``.
        dtype (torch.dtype): Model dtype for meta-init and ``from_pretrained``.
            Training models typically pass ``torch.float32`` (FSDP mixed-precision handles cast);
            ref/inference models pass ``torch.bfloat16`` or ``torch.float16``.
        device_map (str | None): Device placement for model weights. Use ``"meta"``
            to create structure-only meta tensors; use ``"auto"`` for automatic
            placement with Accelerate; use ``"cuda:{rank}"`` to place the whole
            model on a specific GPU.
        use_liger_kernel (bool): Use Liger Kernel optimised causal LM. Defaults to ``False``.
        packing_samples (bool): Pack samples during training. Defaults to ``False``.
        temperature (float): Temperature for log-prob computation. Defaults to 1.0.
    """

    def __init__(
        self,
        pretrain: str,
        *,
        attn_implementation="flash_attention_2",
        torch_dtype: torch.dtype = torch.float32,
        device_map: Optional[str] = "meta",
        packing_samples=False,
        temperature=1.0,
        use_liger_kernel=False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.packing_samples = packing_samples

        if device_map == "meta":
            # -- Build causal-LM structure on meta device (no weight allocation) --
            cfg = AutoConfig.from_pretrained(pretrain, trust_remote_code=True)
            cfg.use_cache = False
            cfg._attn_implementation = attn_implementation

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
        else:
            # -- Load real weights via from_pretrained --
            model_cls = AutoModelForCausalLM
            if use_liger_kernel:
                from liger_kernel.transformers import AutoLigerKernelForCausalLM

                model_cls = AutoLigerKernelForCausalLM

            load_kwargs = {
                "dtype": torch_dtype,
                "attn_implementation": attn_implementation,
                "trust_remote_code": True,
            }
            if device_map is not None:
                load_kwargs["device_map"] = device_map

            model = model_cls.from_pretrained(pretrain, **load_kwargs)

            self.model = model
            if hasattr(self.model, "config") and hasattr(self.model.config, "use_cache"):
                self.model.config.use_cache = False

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

        log_probs = compute_token_log_probs(output["logits"], rolled_sequences, temperature=self.temperature)

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
