import inspect
from contextlib import nullcontext
from importlib.util import find_spec
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor

from openrlhf.utils.fsdp.packing import (
    cp_dtensor_full_sequence,
    is_automodel_custom_model,
    log_probs_from_vocab_parallel_logits,
    pack_padded_batch,
    pad_to_cp_multiple,
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


_HF_ATTN_IMPLEMENTATIONS = {"eager", "sdpa", "flash_attention_2", "flash_attention_3", "te"}
_CUSTOM_ATTN_IMPLEMENTATIONS = {"te", "sdpa", "flex"}
_ALL_ATTN_IMPLEMENTATIONS = _HF_ATTN_IMPLEMENTATIONS | _CUSTOM_ATTN_IMPLEMENTATIONS


def _validate_attn_implementation(attn_implementation: str) -> None:
    if attn_implementation not in _ALL_ATTN_IMPLEMENTATIONS:
        choices = ", ".join(sorted(_ALL_ATTN_IMPLEMENTATIONS))
        raise ValueError(f"Unsupported attention implementation {attn_implementation!r}; choose one of: {choices}")
    if attn_implementation == "te" and find_spec("transformer_engine") is None:
        raise ValueError("--fsdp.attn_implementation te requires transformer-engine to be installed.")


def _uses_cpu_offload(distributed_config) -> bool:
    return getattr(distributed_config, "offload_policy", None) is not None


def _move_model_to_cpu_for_offload(model: nn.Module, distributed_config):
    if not _uses_cpu_offload(distributed_config):
        return model
    for buffer in model.buffers():
        buffer.data = buffer.data.to("cpu")
    return model.to("cpu")


def _patch_nemo_llama_rope_position_ids() -> None:
    """Make NeMo AutoModel Llama RoPE honor non-trivial position_ids.

    Current AutoModel main still slices RoPE caches by sequence length. PPO/RM
    batches use padded ``position_ids``, so gather by id to match HF/vLLM.
    """
    try:
        from nemo_automodel.components.models.llama import rope_utils
    except Exception:
        return

    cls = getattr(rope_utils, "LlamaRotaryEmbedding", None)
    if cls is None or getattr(cls, "_openrlhf_position_ids_patch", False):
        return

    @torch.no_grad()
    def _forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        position_ids = position_ids.to(device=x.device, dtype=torch.long)
        cache_len = position_ids.shape[-1]
        if position_ids.numel() > 0:
            cache_len = max(cache_len, int(position_ids.max().item()) + 1)

        if self._cos_cache is None or cache_len > self.max_seq_len_cached or self._cos_cache.device != x.device:
            self._build_cache(cache_len, x.device)

        flat_positions = position_ids.reshape(-1)
        cos = self._cos_cache.index_select(0, flat_positions).view(*position_ids.shape, -1)
        sin = self._sin_cache.index_select(0, flat_positions).view(*position_ids.shape, -1)

        if self.rope_fusion:
            arange_positions = torch.arange(position_ids.shape[-1], device=position_ids.device).unsqueeze(0)
            if torch.equal(position_ids, arange_positions.expand_as(position_ids)):
                return cos, sin, self._freqs_cache[: position_ids.shape[-1]]

        return cos, sin

    cls.forward = _forward
    cls._openrlhf_position_ids_patch = True


def _patch_nemo_float32_rms_norm_compile() -> None:
    """Run AutoModel fp32 RMSNorm eagerly on dynamic PPO batches.

    AutoModel currently compiles this tiny function with dynamic shapes; TP/FSDP
    PPO can hit an Inductor assertion. The eager replacement keeps the same math.
    """
    try:
        from nemo_automodel.components.models.common import utils as nemo_utils
    except Exception:
        return

    cls = getattr(nemo_utils, "Float32RMSNorm", None)
    if cls is None or getattr(cls, "_openrlhf_eager_patch", False):
        return

    def _float32_rms_norm_fwd(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        return (weight * x).to(input_dtype)

    def _forward(self, x: torch.Tensor):
        return _float32_rms_norm_fwd(x, self.weight, self.eps)

    nemo_utils._float32_rms_norm_fwd = _float32_rms_norm_fwd
    cls.forward = _forward
    cls._openrlhf_eager_patch = True


def _patch_nemo_automodel_runtime() -> None:
    """Apply small upstream-compatibility fixes needed by custom AutoModel paths."""
    _patch_nemo_llama_rope_position_ids()
    _patch_nemo_float32_rms_norm_compile()


def _resolve_custom_backend_attn(attn_implementation: str, packing_samples: bool) -> str:
    if packing_samples:
        if attn_implementation == "te":
            return "te"
        if attn_implementation == "flash_attention_2":
            raise ValueError(
                "--fsdp.packing_samples with AutoModel custom models requires --fsdp.attn_implementation te. "
                "To use flash_attention_2 packing, route through the HF fallback with --fsdp.force_hf_model."
            )
        raise ValueError("--fsdp.packing_samples supports only --fsdp.attn_implementation te " "or flash_attention_2.")

    if attn_implementation in _CUSTOM_ATTN_IMPLEMENTATIONS:
        return attn_implementation

    print(f"[Attn] AutoModel custom models do not use {attn_implementation}; using sdpa backend.")
    return "sdpa"


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


def _class_source_supports_thd_packing(model_cls) -> bool:
    try:
        source_path = inspect.getsourcefile(model_cls)
    except (TypeError, OSError):
        return False
    if not source_path:
        return False
    try:
        source = Path(source_path).read_text(errors="ignore")
    except OSError:
        return False
    return "qkv_format" in source and "cu_seqlens" in source


def _automodel_arch_supports_thd_packing(pretrain_or_model) -> bool:
    """Return whether AutoModel's custom class consumes THD packing kwargs."""
    if not isinstance(pretrain_or_model, str):
        return _automodel_custom_supports_thd_packing(pretrain_or_model)
    try:
        from nemo_automodel._transformers.registry import ModelRegistry
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(pretrain_or_model, trust_remote_code=True)
        archs = getattr(cfg, "architectures", None) or []
        if not archs:
            return False
        model_cls = ModelRegistry.model_arch_name_to_cls.get(archs[0])
        return bool(model_cls) and _class_source_supports_thd_packing(model_cls)
    except Exception:
        return False


def _automodel_custom_supports_thd_packing(model: nn.Module) -> bool:
    if not is_automodel_custom_model(model):
        return False
    return any(_class_source_supports_thd_packing(cls) for cls in type(model).__mro__)


def _build_peft_config_dict(rank: int, alpha: int, dropout: float, target_modules):
    """Map OpenRLHF lora.* args onto AutoModel's PeftConfig dataclass.

    Field-name gotcha: AutoModel renames `r` (LoRA rank) to `dim`.
    Returns a ``PeftConfig`` instance; Automodel's downstream
    ``apply_lora_to_linear_modules`` does attribute access on the config, so a
    plain dict trips ``AttributeError: 'dict' object has no attribute ...``.
    """
    from nemo_automodel.components._peft.lora import PeftConfig

    if isinstance(target_modules, str):
        target_modules = [target_modules]
    base = {"dim": rank, "alpha": alpha, "dropout": dropout}
    # Map HF-peft sentinel "all-linear" to AutoModel's `match_all_linear=True`.
    if not target_modules or target_modules == ["all-linear"]:
        return PeftConfig.from_dict({**base, "match_all_linear": True})
    return PeftConfig.from_dict({**base, "target_modules": list(target_modules)})


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

    Builds the underlying model via AutoModel's official entry
    (``NeMoAutoModelForCausalLM.from_pretrained`` / ``NeMoAutoModelForImageTextToText``),
    which in a single call: loads HF weights, applies the per-architecture TP plan,
    wraps with FSDP2 over ``device_mesh``, attaches CP hooks if cp_size>1, and
    optionally applies LoRA + activation checkpointing.
    """

    def __init__(
        self,
        pretrain_or_model,
        attn_implementation: str = "flash_attention_2",
        param_dtype: str = "bf16",
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
        force_hf_model: bool = False,
        temperature: float = 1.0,
        use_liger_kernel: bool = False,
        freeze_visual_encoder: bool = False,
        use_fp32_master_weights: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.packing_samples = packing_samples
        self._forward_autocast_dtype = None
        self.device_mesh = device_mesh
        mesh_dims = getattr(device_mesh, "mesh_dim_names", ()) or ()
        self.cp_mesh = device_mesh["cp"] if device_mesh is not None and "cp" in mesh_dims else None
        self.cp_size = self.cp_mesh.size() if self.cp_mesh is not None else 1

        if not isinstance(pretrain_or_model, str):
            self.model = pretrain_or_model
            self.is_vlm = False
            self._packing_style = "automodel" if is_automodel_custom_model(self.model) else "hf"
            if self.packing_samples and self._packing_style == "automodel":
                if not _automodel_custom_supports_thd_packing(self.model):
                    raise ValueError(
                        "This pre-instantiated AutoModel custom model does not consume THD packing kwargs. "
                        "Load from a checkpoint path so Actor can fall back to HF FlashAttention2 packing, "
                        "or disable --fsdp.packing_samples."
                    )
            if self.packing_samples and self._packing_style == "hf":
                cfg = getattr(self.model, "config", None)
                if getattr(cfg, "_attn_implementation", None) != "flash_attention_2" or not _has_hf_flash_attn_2():
                    raise ValueError(
                        "HF packed sequence requires flash_attention_2 and flash-attn. "
                        "Use an AutoModel custom TE model or load the HF model with flash_attention_2."
                    )
            return

        from openrlhf.utils.utils import convert_to_torch_dtype, ensure_torchvision_nms_stub, is_vlm_model

        ensure_torchvision_nms_stub()

        # Trainable actor/critic models keep fp32 master weights; ref/reward can
        # opt into compute dtype to save memory. FSDP2 handles bf16 fwd/bwd via
        # MixedPrecisionPolicy. MoE checkpoints (Qwen3-MoE, GLM-MoE, ...) are a
        # special case: AutoModel PR #1896 fixes fp32-master handling for custom
        # MoE, but that PR is not in the pinned main commit yet. Until then keep
        # MoE params in compute dtype to avoid mixed-dtype FSDP / grouped-GEMM
        # failures.
        compute_dtype = convert_to_torch_dtype(param_dtype)
        is_moe = _detect_moe_arch(pretrain_or_model)
        ep_active = moe_mesh is not None
        auto_force_hf = force_hf_model or (is_moe and not ep_active)
        if (
            packing_samples
            and not auto_force_hf
            and not _will_use_hf_model(pretrain_or_model, False)
            and not _automodel_arch_supports_thd_packing(pretrain_or_model)
        ):
            print(
                "[Packing] AutoModel custom implementation for this architecture does not consume THD packing "
                "kwargs; using HF FlashAttention2 varlen packed path."
            )
            auto_force_hf = True
            attn_implementation = "flash_attention_2"

        _validate_attn_implementation(attn_implementation)
        torch_dtype = compute_dtype if not use_fp32_master_weights else torch.float32
        if is_moe and torch_dtype == torch.float32:
            torch_dtype = compute_dtype
        self.is_vlm = is_vlm_model(pretrain_or_model)

        if self.is_vlm and use_liger_kernel:
            raise ValueError(
                "use_liger_kernel is not compatible with VLM models. "
                "Liger kernel only supports CausalLM, not ImageTextToText."
            )
        # Heads-up before construction: AutoModel disables Liger under TP/CP and
        # for its own custom (non-HF) model implementations. Without this notice
        # the user's --fsdp.use_liger_kernel flag silently no-ops. See
        # nemo_automodel/_transformers/kernel_patches.py:_apply_preload_overrides
        # and auto_model.py "if use_liger_kernel and not is_custom_model".
        if use_liger_kernel:
            mesh_dims = getattr(device_mesh, "mesh_dim_names", ()) or ()
            tp_size = device_mesh["tp"].size() if "tp" in mesh_dims else 1
            cp_size = device_mesh["cp"].size() if "cp" in mesh_dims else 1
            if tp_size > 1 or cp_size > 1:
                print(
                    f"[Liger] AutoModel disables Liger Kernel when TP>1 ({tp_size}) "
                    f"or CP>1 ({cp_size}); --fsdp.use_liger_kernel will be a no-op."
                )

        peft_config = None
        if lora_rank > 0:
            peft_config = _build_peft_config_dict(lora_rank, lora_alpha, lora_dropout, target_modules)
        # Stash on the wrapper so the strategy's save_model path can forward it
        # to AutoModel's Checkpointer; the PEFT addon needs the original config
        # to write adapter_config.json.
        self.peft_config = peft_config

        if self.is_vlm:
            from nemo_automodel import NeMoAutoModelForImageTextToText as ModelCls
        else:
            from nemo_automodel import NeMoAutoModelForCausalLM as ModelCls

        # force_hf=True selection: AutoModel's custom impls have known issues
        # we need to route around:
        #  - User override: keep a manual escape hatch for models whose native
        #    AutoModel path regresses under a given torch/transformers combo.
        #  - MoE without EP: AutoModel's custom MoE requires a non-None
        #    moe_mesh with an 'ep' dim ('AssertionError: ep mesh dimension not
        #    found'). Use HF reference path.
        #  AutoModel custom Qwen3 MoE currently does not advertise a TP plan,
        #  so TP+EP is intentionally left to AutoModel validation.
        force_hf = auto_force_hf
        # Downgrade flash_attention_2 to sdpa for non-packing runs when
        # flash-attn is unavailable. Packing with flash_attention_2 cannot be
        # downgraded because SDPA would not preserve packed sequence
        # boundaries.
        if attn_implementation == "flash_attention_2" and not _has_hf_flash_attn_2():
            if packing_samples:
                raise ValueError("--fsdp.packing_samples with flash_attention_2 requires flash-attn to be installed.")
            print("[Attn] flash_attn not installed; downgrading flash_attention_2 to sdpa.")
            attn_implementation = "sdpa"
        use_hf_model = _will_use_hf_model(pretrain_or_model, force_hf)
        if use_hf_model and attn_implementation == "flex":
            raise ValueError(
                "--fsdp.attn_implementation flex is only supported on AutoModel custom models; "
                "drop --fsdp.force_hf_model or use sdpa/eager/flash_attention_2."
            )
        # Keep AutoModel custom forwards in the compute dtype when master
        # weights are fp32, so lm_head/score inputs match bf16 parameters.
        if compute_dtype != torch.float32 and not (is_moe and not use_hf_model):
            self._forward_autocast_dtype = compute_dtype
        if packing_samples and use_hf_model and attn_implementation != "flash_attention_2":
            raise ValueError(
                "HF packed sequence requires --fsdp.attn_implementation flash_attention_2. "
                "Use --fsdp.attn_implementation te for AutoModel custom THD packing."
            )

        automodel_backend_kwargs = {}
        automodel_attn_implementation = attn_implementation
        automodel_has_packed_sequence = packing_samples
        self._packing_style = "hf" if use_hf_model else "automodel"
        if not use_hf_model:
            _patch_nemo_automodel_runtime()
            backend_attn = _resolve_custom_backend_attn(attn_implementation, packing_samples)
            # AutoModel still validates the HF config-side attention name.
            # The real custom backend is supplied through BackendConfig below.
            automodel_attn_implementation = "sdpa"
            from nemo_automodel.components.models.common.utils import BackendConfig

            using_te = backend_attn == "te"
            backend_kwargs = {"attn": backend_attn, "rope_fusion": using_te}
            if not using_te:
                backend_kwargs["linear"] = "torch"
                backend_kwargs["experts"] = "torch_mm"
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                print(f"[Attn] AutoModel custom backend={backend_attn}; config attn_implementation=sdpa.")
            automodel_backend_kwargs["backend"] = BackendConfig(**backend_kwargs)

        self.model = ModelCls.from_pretrained(
            pretrain_or_model,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            attn_implementation=automodel_attn_implementation,
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
        self.model = _move_model_to_cpu_for_offload(self.model, distributed_config)
        self._packing_style = "automodel" if is_automodel_custom_model(self.model) else "hf"
        if self.packing_samples and self._packing_style == "hf" and attn_implementation != "flash_attention_2":
            raise ValueError("HF packed sequence requires flash_attention_2.")
        if self.packing_samples:
            path = "AutoModel THD/TE" if self._packing_style == "automodel" else "HF FlashAttention2 varlen"
            print(f"[Packing] Using {path} packed path.")

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
        cp_local_log_probs=False,
        cp_context_stack=None,
        packed_seq_lens: Optional[list[int]] = None,
        return_entropy=False,
        **mm_inputs,
    ) -> torch.Tensor:
        """Returns action log probs."""
        batch, seqlen = sequences.size()
        attn_kwargs: dict = {}
        indices = None
        cp_forward = False
        cp_ctx_factory = nullcontext
        if self.packing_samples:
            sequences, position_ids, rolled_sequences, indices, attn_kwargs = pack_padded_batch(
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
                if mm_inputs:
                    cfg = self._vlm_config
                    token_type_ids = (sequences == cfg.image_token_id).to(torch.int32)
                    if getattr(cfg, "video_token_id", None) is not None:
                        token_type_ids[sequences == cfg.video_token_id] = 2
                    key = "mm_token_type_ids" if "image_grid_thw" in mm_inputs else "token_type_ids"
                    mm_inputs[key] = token_type_ids
            elif position_ids is None:
                if attention_mask is None:
                    position_ids = torch.arange(seqlen, device=sequences.device).unsqueeze(0).expand(batch, -1)
                else:
                    position_ids = attention_mask.long().cumsum(-1) - 1
                    position_ids.masked_fill_(attention_mask == 0, 1)

            if self.cp_size > 1 and attention_mask is not None:
                if mm_inputs:
                    raise NotImplementedError("VLM inputs are not supported with --fsdp.cp_size > 1.")
                from nemo_automodel.components.distributed.cp_utils import make_cp_batch_and_ctx

                sequences = pad_to_cp_multiple(sequences, self.cp_size, seq_dim=1, value=0)
                attention_mask = pad_to_cp_multiple(attention_mask, self.cp_size, seq_dim=1, value=0)
                rolled_sequences = pad_to_cp_multiple(rolled_sequences, self.cp_size, seq_dim=1, value=0)
                position_ids = pad_to_cp_multiple(position_ids, self.cp_size, seq_dim=1, value=1)
                cp_batch = {
                    "input_ids": sequences,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "labels": rolled_sequences,
                }
                cp_ctx_factory, cp_batch = make_cp_batch_and_ctx(self.device_mesh, cp_batch)
                sequences = cp_batch["input_ids"]
                position_ids = cp_batch.get("position_ids")
                rolled_sequences = cp_batch["labels"]
                forward_attention_mask = cp_batch.get("attention_mask")
                cp_forward = True

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=self._forward_autocast_dtype)
            if self._forward_autocast_dtype is not None and sequences.is_cuda
            else nullcontext()
        )

        forward_ctx = cp_ctx_factory()
        if cp_context_stack is not None and cp_forward:
            # AutoModel CP train context installs backward hooks, so training
            # code keeps it alive until loss.backward() completes.
            cp_context_stack.enter_context(forward_ctx)
            forward_ctx = nullcontext()

        with forward_ctx:
            with autocast_ctx:
                output = self.model(
                    sequences,
                    attention_mask=forward_attention_mask,
                    position_ids=position_ids,
                    **attn_kwargs,
                    **mm_inputs,
                )
        # AutoModel's custom MoE/LLM models (e.g. Qwen3MoeForCausalLM) return a
        # raw logits Tensor; HF returns a ModelOutput with `.logits`. Normalize.
        output = _normalize_output(output)
        logits = output["logits"]
        full_logits = None

        if return_entropy:
            assert return_output
            full_logits = unshard_dtensor(logits).to(torch.float32)
            if cp_forward and not isinstance(logits, DTensor):
                full_logits = cp_dtensor_full_sequence(full_logits, self.cp_mesh, seq_dim=1)
            if cp_forward:
                full_logits = full_logits[:, :seqlen]
            output["logits"] = full_logits
            entropy = compute_entropy(full_logits)
            if self.packing_samples:
                entropy = unpack_to_padded(entropy, indices, batch, seqlen)
            output.entropy = entropy[:, :-1]

        return_action_log_probs = action_mask is not None
        if cp_forward and cp_local_log_probs and return_action_log_probs:
            raise ValueError("cp_local_log_probs returns local sequence logprobs; pass action_mask=None.")
        if not return_action_log_probs and not return_logprobs:
            assert return_output
            if allgather_logits:
                full_logits = full_logits if full_logits is not None else unshard_dtensor(logits).to(torch.float32)
                if cp_forward and not isinstance(logits, DTensor):
                    full_logits = cp_dtensor_full_sequence(full_logits, self.cp_mesh, seq_dim=1)
                if cp_forward:
                    full_logits = full_logits[:, :seqlen]
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

        if cp_forward and not cp_local_log_probs:
            log_probs = cp_dtensor_full_sequence(log_probs, self.cp_mesh, seq_dim=1)
            log_probs = log_probs[:, :seqlen]

        if self.packing_samples:
            log_probs = unpack_to_padded(log_probs, indices, batch, seqlen)

        if truncate_logprobs and not (cp_forward and cp_local_log_probs):
            log_probs = log_probs[:, :-1]
        if not return_action_log_probs and return_logprobs:
            return (log_probs, output) if return_output else log_probs

        action_log_probs = log_probs[:, -action_mask.shape[1] :] * action_mask.float()
        return (action_log_probs, output) if return_output else action_log_probs

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        # No-op under FSDP2/AutoModel: activation checkpointing is configured
        # at construction time via `activation_checkpointing=True` on
        # `NeMoAutoModelForCausalLM.from_pretrained`. Calling HF's late hook
        # would conflict with FSDP2's already-applied wrap.
        pass

    def gradient_checkpointing_disable(self):
        pass

    def print_trainable_parameters(self):
        if hasattr(self.model, "print_trainable_parameters"):
            self.model.print_trainable_parameters()
