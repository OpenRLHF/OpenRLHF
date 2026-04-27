from contextlib import nullcontext
from importlib.util import find_spec
from typing import Optional

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor
from transformers import BitsAndBytesConfig

from openrlhf.utils.fsdp.packing import (
    is_automodel_custom_model,
    log_probs_from_vocab_parallel_logits,
    pack_padded_batch,
    unpack_to_padded,
    unshard_dtensor,
)

from .utils import compute_entropy, log_probs_from_logits


def _detect_moe_arch(pretrain_or_model) -> bool:
    """Lightweight MoE detection from HF config (no model load)."""
    if not isinstance(pretrain_or_model, str):
        return False
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(pretrain_or_model, trust_remote_code=True)
        archs = getattr(cfg, "architectures", None) or []
        if any("Moe" in a or "MoE" in a for a in archs):
            return True
        for k in ("num_experts", "n_routed_experts", "num_local_experts", "moe_num_experts"):
            n = getattr(cfg, k, None)
            if isinstance(n, int) and n > 1:
                return True
    except Exception:
        return False
    return False


def _has_hf_flash_attn_2() -> bool:
    try:
        from transformers.utils import is_flash_attn_2_available

        return bool(is_flash_attn_2_available())
    except Exception:
        return find_spec("flash_attn") is not None


def _will_use_hf_model(pretrain_or_model, force_hf: bool, default: bool = True) -> bool:
    if force_hf:
        return True
    if not isinstance(pretrain_or_model, str):
        return False
    try:
        from nemo_automodel._transformers.model_init import get_is_hf_model
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(pretrain_or_model, trust_remote_code=True)
        return get_is_hf_model(cfg, force_hf)
    except Exception:
        return default


def _hf_packing_supported(attn_implementation: str) -> bool:
    # HF packed sequence support is the flash-attn2 varlen path. Other
    # backends would treat the packed stream as one long sequence and leak
    # attention across sample boundaries.
    return attn_implementation == "flash_attention_2" and _has_hf_flash_attn_2()


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


class _AttrDict(dict):
    """Dict output that also supports ``output.foo`` trainer access."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


def _normalize_output(output):
    if isinstance(output, torch.Tensor):
        return _AttrDict(logits=output)
    if isinstance(output, dict) and not isinstance(output, _AttrDict):
        return _AttrDict(output)
    return output


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
        moe_mesh=None,
        distributed_config=None,
        moe_config=None,
        activation_checkpointing: bool = False,
        packing_samples: bool = False,
        temperature: float = 1.0,
        use_liger_kernel: bool = False,
        freeze_visual_encoder: bool = False,
        use_fp32_master_weights: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.packing_samples = packing_samples
        self._packing_style = "hf"
        self._forward_autocast_dtype = None

        if not isinstance(pretrain_or_model, str):
            self.model = pretrain_or_model
            self.is_vlm = False
            self._packing_style = "automodel" if is_automodel_custom_model(self.model) else "hf"
            return

        from openrlhf.utils.utils import convert_to_torch_dtype, ensure_torchvision_nms_stub, is_vlm_model

        # Trainable actor/critic models keep fp32 master weights; ref/reward can
        # opt into compute dtype to save memory. FSDP2 handles bf16 fwd/bwd via
        # MixedPrecisionPolicy. MoE checkpoints (Qwen3-MoE, GLM-MoE, ...) carry
        # mixed bf16 expert weights + fp32 router gates; loading with
        # torch_dtype=fp32 leaves the expert side as bf16, and FSDP rejects
        # ('expects uniform original parameter dtype'). Drop fp32 master for
        # MoE on the HF reference path (Automodel custom MoE handles the mix
        # internally and works with fp32 master).
        compute_dtype = convert_to_torch_dtype(param_dtype)
        is_moe = _detect_moe_arch(pretrain_or_model)
        torch_dtype = compute_dtype if load_in_4bit or not use_fp32_master_weights else torch.float32
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
                bnb_4bit_compute_dtype=compute_dtype,
            )

        peft_config = None
        if lora_rank > 0:
            peft_config = _build_peft_config_dict(lora_rank, lora_alpha, lora_dropout, target_modules)

        ensure_torchvision_nms_stub()
        if self.is_vlm:
            from nemo_automodel import NeMoAutoModelForImageTextToText as ModelCls
        else:
            from nemo_automodel import NeMoAutoModelForCausalLM as ModelCls

        # force_hf=True selection — Automodel's custom impls have known issues
        # we need to route around:
        #  - Dense + TP>1 + EP=1: F.linear non-contiguous-DTensor view error.
        #    Use HF reference TP path.
        #  - MoE without EP: Automodel's custom MoE requires a non-None
        #    moe_mesh with an 'ep' dim ('AssertionError: ep mesh dimension not
        #    found'). Use HF reference path.
        #  Automodel custom Qwen3 MoE currently does not advertise a TP plan,
        #  so TP+EP is intentionally left to Automodel validation.
        tp_size = device_mesh["tp"].size() if device_mesh is not None and "tp" in device_mesh.mesh_dim_names else 1
        ep_active = moe_mesh is not None
        force_hf = (tp_size > 1 and not ep_active) or (is_moe and not ep_active)
        # MoE on HF path: see comment near torch_dtype — drop fp32 master to
        # avoid the mixed-dtype FSDP assert.
        if is_moe and force_hf and torch_dtype == torch.float32:
            torch_dtype = compute_dtype
        if torch_dtype == torch.float32 and attn_implementation in ("flash_attention_2", "flash_attention_3"):
            self._forward_autocast_dtype = compute_dtype
        use_hf_model = _will_use_hf_model(pretrain_or_model, force_hf)
        automodel_has_packed_sequence = packing_samples
        if packing_samples and use_hf_model and not _hf_packing_supported(attn_implementation):
            print(
                "[Packing] HF fallback packed sequence requires flash_attention_2 + flash_attn. "
                "Disabling packing for this model and using padded forward instead."
            )
            automodel_has_packed_sequence = False
            self.packing_samples = False
            if attn_implementation == "flash_attention_2" and not _has_hf_flash_attn_2():
                attn_implementation = "sdpa"

        automodel_backend_kwargs = {}
        if attn_implementation in {"te", "sdpa", "flex"} and not use_hf_model:
            from nemo_automodel.components.models.common.utils import BackendConfig

            automodel_backend_kwargs["backend"] = BackendConfig(attn=attn_implementation)

        self.model = ModelCls.from_pretrained(
            pretrain_or_model,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            quantization_config=nf4_config,
            device_mesh=device_mesh,
            moe_mesh=moe_mesh,
            distributed_config=distributed_config,
            moe_config=moe_config,
            activation_checkpointing=activation_checkpointing,
            peft_config=peft_config,
            use_liger_kernel=use_liger_kernel and not self.is_vlm,
            has_packed_sequence=automodel_has_packed_sequence,
            force_hf=force_hf,
            **automodel_backend_kwargs,
        )
        self._packing_style = "automodel" if is_automodel_custom_model(self.model) else "hf"
        if self.packing_samples and self._packing_style == "hf" and not _hf_packing_supported(attn_implementation):
            print(
                "[Packing] Loaded model is HF fallback without flash_attention_2 packed support. "
                "Disabling packing for correctness."
            )
            self.packing_samples = False

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
        position_ids: Optional[torch.Tensor] = None,
        logprob_labels: Optional[torch.Tensor] = None,
        truncate_logprobs: bool = True,
        return_output=False,
        allgather_logits=False,
        return_logprobs=False,
        packed_seq_lens: Optional[list[int]] = None,
        return_entropy=False,
        **mm_inputs,
    ) -> torch.Tensor:
        """Returns action log probs."""
        batch, seqlen = sequences.size()
        fa_kwargs: dict = {}
        indices = None
        if self.packing_samples:
            sequences, position_ids, rolled_sequences, indices, fa_kwargs = pack_padded_batch(
                sequences, attention_mask, style=self._packing_style
            )
            forward_attention_mask = None
        else:
            # https://github.com/OpenRLHF/OpenRLHF/issues/217
            rolled_sequences = (
                logprob_labels if logprob_labels is not None else torch.roll(sequences, shifts=-1, dims=1)
            )
            forward_attention_mask = attention_mask

            if getattr(self, "is_vlm", False):
                if position_ids is None:
                    position_ids = None
                if mm_inputs:
                    cfg = self._vlm_config
                    token_type_ids = (sequences == cfg.image_token_id).to(torch.int32)
                    if getattr(cfg, "video_token_id", None) is not None:
                        token_type_ids[sequences == cfg.video_token_id] = 2
                    key = "mm_token_type_ids" if "image_grid_thw" in mm_inputs else "token_type_ids"
                    mm_inputs[key] = token_type_ids
            else:
                if position_ids is None:
                    if attention_mask is None:
                        position_ids = torch.arange(seqlen, device=sequences.device).unsqueeze(0).expand(batch, -1)
                    else:
                        position_ids = attention_mask.long().cumsum(-1) - 1
                        position_ids.masked_fill_(attention_mask == 0, 1)

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=self._forward_autocast_dtype)
            if self._forward_autocast_dtype is not None and sequences.is_cuda
            else nullcontext()
        )
        with autocast_ctx:
            output = self.model(
                sequences,
                attention_mask=forward_attention_mask,
                position_ids=position_ids,
                **fa_kwargs,
                **mm_inputs,
            )
        # Automodel's custom MoE/LLM models (e.g. Qwen3MoeForCausalLM) return a
        # raw logits Tensor; HF returns a ModelOutput with `.logits`. Normalize.
        output = _normalize_output(output)
        logits = output["logits"]
        full_logits = None

        if return_entropy:
            assert return_output
            full_logits = unshard_dtensor(logits).to(torch.float32)
            output["logits"] = full_logits
            entropy = compute_entropy(full_logits)
            if self.packing_samples:
                entropy = unpack_to_padded(entropy, indices, batch, seqlen)
            output.entropy = entropy[:, :-1]

        return_action_log_probs = action_mask is not None
        if not return_action_log_probs and not return_logprobs:
            assert return_output
            if allgather_logits:
                full_logits = full_logits if full_logits is not None else unshard_dtensor(logits).to(torch.float32)
                output["logits"] = (
                    unpack_to_padded(full_logits, indices, batch, seqlen) if self.packing_samples else full_logits
                )
            return output

        if isinstance(logits, DTensor):
            log_probs = log_probs_from_vocab_parallel_logits(
                logits,
                rolled_sequences,
                temperature=self.temperature,
            )
        else:
            full_logits = full_logits if full_logits is not None else logits.to(torch.float32)
            log_probs = log_probs_from_logits(full_logits, rolled_sequences, temperature=self.temperature)

        if self.packing_samples:
            log_probs = unpack_to_padded(log_probs, indices, batch, seqlen)

        if truncate_logprobs:
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
