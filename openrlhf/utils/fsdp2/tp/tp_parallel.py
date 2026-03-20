"""
Tensor Parallelism for FSDP2
============================

Applies DTensor-based TP to HuggingFace causal LM models.

Plan resolution order: custom_plan > HF model._tp_plan > default.
Each plan maps module-name glob patterns to ParallelStyle instances.
"""

from __future__ import annotations

import logging
from fnmatch import fnmatch
from functools import partial

import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)
from torch.distributed.tensor.parallel.style import ParallelStyle, distribute_module
from torch.distributed.tensor.placement_types import Placement

from ..utils import ensure_tied_word_embeddings

logger = logging.getLogger(__name__)


# =============================================================================
# Custom ParallelStyle classes
# =============================================================================


class ReplicateParallel(ParallelStyle):
    """Replicate parameters without sharding, with custom I/O layout control.

    Upstream PyTorch only has sharding styles (Colwise/Rowwise/SequenceParallel).
    This fills the gap for modules that should NOT be sharded (e.g. score head,
    Q/K RMSNorm) but still need DTensor metadata for FSDP2 grad norm & checkpointing.
    """

    def __init__(
        self,
        *,
        input_layout: Placement | None = None,
        desired_input_layout: Placement | None = None,
        output_layout: Placement | None = None,
        use_local_output: bool = True,
    ):
        super().__init__()
        self.input_layout = input_layout or Replicate()
        self.output_layout = output_layout or Replicate()
        self.desired_input_layout = desired_input_layout or Replicate()
        self.use_local_output = use_local_output

    @staticmethod
    def _prepare_input_fn(input_layout, desired_input_layout, mod, inputs, device_mesh):
        input_tensor = inputs[0]
        if not isinstance(input_tensor, DTensor):
            input_tensor = DTensor.from_local(input_tensor, device_mesh, (input_layout,), run_check=False)
        if input_layout != desired_input_layout:
            input_tensor = input_tensor.redistribute(placements=(desired_input_layout,), async_op=True)
        return (input_tensor, *inputs[1:])

    @staticmethod
    def _prepare_output_fn(output_layout, use_local_output, mod, outputs, device_mesh):
        if outputs.placements != (output_layout,):
            outputs = outputs.redistribute(placements=(output_layout,), async_op=True)
        return outputs.to_local() if use_local_output else outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            None,
            partial(self._prepare_input_fn, self.input_layout, self.desired_input_layout),
            partial(self._prepare_output_fn, self.output_layout, self.use_local_output),
        )


class SequenceParallelPreserveGrad(SequenceParallel):
    """SequenceParallel that preserves requires_grad when wrapping parameters.

    Upstream SequenceParallel._replicate_module_fn wraps params via
    nn.Parameter(dtensor) which resets requires_grad=True, silently unfreezing
    frozen params. Critical for LoRA / partial fine-tuning.
    """

    def _replicate_module_fn(self, name: str, module: nn.Module, device_mesh: DeviceMesh) -> None:
        for p_name, param in module.named_parameters():
            module.register_parameter(
                p_name,
                nn.Parameter(
                    DTensor.from_local(param, device_mesh, [Replicate()], run_check=False),
                    requires_grad=param.requires_grad,
                ),
            )


# =============================================================================
# TP plan building
# =============================================================================

# Attention + MLP projections (covers both separate and fused variants)
_ATTN_MLP_PLAN: dict[str, ParallelStyle] = {
    "model.layers.*.self_attn.q_proj": ColwiseParallel(),
    "model.layers.*.self_attn.k_proj": ColwiseParallel(),
    "model.layers.*.self_attn.v_proj": ColwiseParallel(),
    "model.layers.*.self_attn.qkv_proj": ColwiseParallel(),  # fused QKV
    "model.layers.*.self_attn.o_proj": RowwiseParallel(),
    "model.layers.*.mlp.gate_proj": ColwiseParallel(),
    "model.layers.*.mlp.up_proj": ColwiseParallel(),
    "model.layers.*.mlp.gate_up_proj": ColwiseParallel(),  # fused gate+up
    "model.layers.*.mlp.down_proj": RowwiseParallel(),
}


def _build_default_tp_plan(sequence_parallel: bool = False) -> dict[str, ParallelStyle]:
    """Build TP plan for LLaMA-style transformers.

    Without SP: embeddings/lm_head use Replicate I/O, attention+MLP use Colwise/Rowwise.
    With SP: activations are Shard(1) between layers, all-gathered before attention/MLP.
    """
    if not sequence_parallel:
        return {
            "model.embed_tokens": RowwiseParallel(input_layouts=Replicate()),
            "lm_head": ColwiseParallel(output_layouts=Replicate(), use_local_output=True),
            **_ATTN_MLP_PLAN,
        }

    # Sequence Parallel: activations sharded on seq dim (Shard(1)) between layers
    return {
        **_ATTN_MLP_PLAN,
        "model.embed_tokens": RowwiseParallel(input_layouts=Replicate(), output_layouts=Shard(1)),
        "model.norm": SequenceParallelPreserveGrad(use_local_output=True),
        "model.layers.*.input_layernorm": SequenceParallelPreserveGrad(use_local_output=True),
        "model.layers.*.post_attention_layernorm": SequenceParallelPreserveGrad(use_local_output=True),
        # All-gather before Attention/MLP: Shard(1) → Replicate → local tensor for flash-attn
        "model.layers.*.self_attn": PrepareModuleInput(
            input_kwarg_layouts={"hidden_states": Shard(1)},
            desired_input_kwarg_layouts={"hidden_states": Replicate()},
            use_local_output=True,
        ),
        "model.layers.*.mlp": PrepareModuleInput(
            input_layouts=Shard(1),
            desired_input_layouts=Replicate(),
            use_local_output=True,
        ),
        # Reduce-scatter back to Shard(1) after Attention/MLP
        "model.layers.*.self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
        "model.layers.*.mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
        # lm_head: Shard(1) input → Replicate output (full vocab logits)
        "lm_head": ColwiseParallel(input_layouts=Shard(1), output_layouts=Replicate(), use_local_output=True),
    }


# =============================================================================
# Plan resolution and modification
# =============================================================================

_STYLE_SHORTHAND = {
    "colwise": lambda: ColwiseParallel(),
    "rowwise": lambda: RowwiseParallel(),
    "colwise_rep": lambda: ColwiseParallel(output_layouts=Replicate()),
    "rowwise_rep": lambda: RowwiseParallel(input_layouts=Replicate()),
    "sequence_parallel": lambda: SequenceParallelPreserveGrad(),
    "replicate": lambda: ReplicateParallel(),
}


def _parse_parallel_style(style_name: str) -> ParallelStyle:
    """Convert string shorthand (e.g. "colwise") to ParallelStyle instance.

    Falls back to HF's ParallelInterface mapping for styles not in our shorthand
    (e.g. "colwise_gather_output" added in Transformers v5).
    """
    if style_name in _STYLE_SHORTHAND:
        return _STYLE_SHORTHAND[style_name]()
    # Fallback: HF Transformers v5+ defines its own style mapping
    try:
        from transformers.integrations.tensor_parallel import ParallelInterface

        hf_mapping = ParallelInterface._global_mapping
        if style_name in hf_mapping:
            return hf_mapping[style_name]
    except (ImportError, AttributeError):
        pass
    raise KeyError(f"Unknown TP style: {style_name!r}. Add it to _STYLE_SHORTHAND or upgrade transformers.")


def _extract_hf_tp_plan(model: nn.Module, sequence_parallel: bool = False) -> dict[str, ParallelStyle] | None:
    """Read _tp_plan from HuggingFace model class/instance and normalize it.

    Searches: type(model)._tp_plan, model._tp_plan, model.model._tp_plan.
    Inner-model entries get "model." prefix. String shorthands are converted.

    If the HF plan is too sparse (e.g. only lm_head), merges with the default
    plan so that attention/MLP projections are still parallelized.
    """
    hf_entries: dict[str, ParallelStyle | str] = {}
    for src in [type(model), model, getattr(model, "model", None)]:
        tp_plan = getattr(src, "_tp_plan", None) if src else None
        if tp_plan:
            prefix = "model." if src == getattr(model, "model", None) else ""
            hf_entries.update({f"{prefix}{k}": v for k, v in tp_plan.items()})
    if not hf_entries:
        return None

    hf_entries.setdefault("model.embed_tokens", "rowwise_rep")
    plan: dict[str, ParallelStyle] = {
        k: _parse_parallel_style(v) if isinstance(v, str) else v for k, v in hf_entries.items()
    }

    # If the HF plan is too sparse (no attention/MLP entries), merge with
    # the default plan so we don't silently skip sharding heavy linear layers.
    has_attn_mlp = any("self_attn" in k or "mlp" in k for k in plan)
    if not has_attn_mlp:
        default = _build_default_tp_plan(sequence_parallel)
        # Default plan provides the base; HF entries override it
        merged = dict(default)
        merged.update(plan)
        plan = merged

    # Always return full-vocab logits (Replicate) for simple loss functions
    if "lm_head" not in plan:
        plan["lm_head"] = ColwiseParallel(output_layouts=Replicate(), use_local_output=True)
    elif isinstance(plan["lm_head"], ColwiseParallel):
        plan["lm_head"] = ColwiseParallel(
            input_layouts=plan["lm_head"].input_layouts[0],
            output_layouts=Replicate(),
            use_local_output=True,
        )

    return plan


def _add_qwen_norm_entries(plan: dict[str, ParallelStyle]) -> dict[str, ParallelStyle]:
    """Add Qwen-specific q_norm/k_norm ReplicateParallel entries.

    These norms operate on head-sharded activations (batch, seq, num_heads, head_dim)
    after the QKV reshape. Shard(2) marks the heads dimension.
    """
    plan = dict(plan)
    for name in ("q_norm", "k_norm"):
        key = f"model.layers.*.self_attn.{name}"
        plan[key] = ReplicateParallel(
            input_layout=Shard(2),
            desired_input_layout=Shard(2),
            output_layout=Shard(2),
            use_local_output=True,
        )
    return plan


# Architecture-specific TP plan augmentations.
# Maps HuggingFace model class name → function that adds extra entries to a plan.
_MODEL_PLAN_AUGMENTATIONS = {
    "Qwen2ForCausalLM": _add_qwen_norm_entries,
    "Qwen3ForCausalLM": _add_qwen_norm_entries,
}


def _with_loss_parallel_lm_head(plan: dict[str, ParallelStyle], sequence_parallel: bool) -> dict[str, ParallelStyle]:
    """Override lm_head to output vocab-sharded DTensor (Shard(-1)) for loss-parallel."""
    plan = dict(plan)
    plan["lm_head"] = ColwiseParallel(
        input_layouts=Shard(1) if sequence_parallel else Replicate(),
        output_layouts=Shard(-1),
        use_local_output=False,
    )
    return plan


def _ensure_value_head_in_plan(
    model: nn.Module,
    plan: dict[str, ParallelStyle],
    sequence_parallel: bool = False,
) -> dict[str, ParallelStyle]:
    """Add ReplicateParallel for score layer if model is Reward/Critic.

    The score head produces per-token scalars — must NOT be vocab-sharded.
    With SP, input is Shard(1) and needs all-gather before the linear.
    """
    head = getattr(model, "score", None)
    if head is None:
        return plan
    if not isinstance(head, nn.Module):
        raise RuntimeError("Value model must expose `score` as an nn.Module.")

    plan = dict(plan)
    input_layout = Shard(1) if sequence_parallel else Replicate()
    plan["score"] = ReplicateParallel(
        input_layout=input_layout,
        desired_input_layout=Replicate(),
        output_layout=Replicate(),
        use_local_output=True,
    )
    logger.info("Added ReplicateParallel for score layer (SP=%s)", sequence_parallel)
    return plan


def _prune_plan(plan: dict[str, ParallelStyle], model: nn.Module) -> dict[str, ParallelStyle]:
    """Drop plan entries that don't match any module (avoids warnings for optional fused layers)."""
    module_names = {name for name, _ in model.named_modules()}
    return {k: v for k, v in plan.items() if any(fnmatch(name, k) for name in module_names)}


# =============================================================================
# Public API
# =============================================================================


def get_tp_plan(
    model: nn.Module,
    sequence_parallel: bool = False,
    shard_logits: bool = False,
    custom_plan: dict[str, ParallelStyle] | None = None,
) -> dict[str, ParallelStyle]:
    """Resolve TP plan: custom_plan > HF model._tp_plan > default LLaMA-style plan.

    Architecture-specific augmentations (e.g. Qwen q_norm/k_norm) are applied
    unless a fully custom plan is provided.
    """
    plan = custom_plan or _extract_hf_tp_plan(model, sequence_parallel) or _build_default_tp_plan(sequence_parallel)

    # Apply architecture-specific augmentations (unless caller provided a full custom plan)
    if not custom_plan:
        augment_fn = _MODEL_PLAN_AUGMENTATIONS.get(type(model).__name__)
        if augment_fn is not None:
            try:
                plan = augment_fn(plan)
            except Exception as e:
                logger.warning("TP plan augmentation failed for %s: %s", type(model).__name__, e)

    if shard_logits:
        plan = _with_loss_parallel_lm_head(plan, sequence_parallel)

    plan = _prune_plan(plan, model)
    if not plan:
        raise ValueError(
            f"No TP plan entries matched any modules (model_cls={type(model).__name__}). "
            "Disable TP (fsdp2_tp_size=1) or ensure the model has _tp_plan."
        )
    return plan


def validate_tp_mesh(model: nn.Module, tp_mesh: DeviceMesh) -> None:
    """Check that attention heads are divisible by TP size."""
    if tp_mesh.size() == 1:
        return
    cfg = getattr(model, "config", None)
    if not cfg:
        return
    tp_size = tp_mesh.size()
    num_heads = getattr(cfg, "num_attention_heads", 0)
    num_kv_heads = getattr(cfg, "num_key_value_heads", 0) or num_heads
    if num_heads and num_heads % tp_size:
        raise ValueError(f"num_attention_heads ({num_heads}) not divisible by TP ({tp_size})")
    if num_kv_heads and num_kv_heads % tp_size:
        raise ValueError(f"num_key_value_heads ({num_kv_heads}) not divisible by TP ({tp_size})")


def maybe_enable_async_tp(tp_mesh: DeviceMesh, enabled: bool = False) -> None:
    """Enable Async TP (Symmetric Memory) for NVLink environments."""
    if not enabled:
        return
    try:
        import torch._inductor.config
        from torch.distributed._symmetric_memory import enable_symm_mem_for_group

        torch._inductor.config._micro_pipeline_tp = True
        enable_symm_mem_for_group(tp_mesh.get_group().group_name)
        logger.info("Enabled Async TP (Symmetric Memory)")
    except Exception as e:
        logger.warning("Failed to enable Async TP: %s", e)


def apply_tensor_parallel(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    sequence_parallel: bool = False,
    validate: bool = True,
    enable_async_tp: bool = False,
    shard_logits: bool = False,
    custom_plan: dict[str, ParallelStyle] | None = None,
) -> nn.Module:
    """Apply Tensor Parallelism to a model. Returns model unchanged if tp_mesh.size() == 1."""
    if tp_mesh.size() == 1:
        return model

    maybe_enable_async_tp(tp_mesh, enabled=enable_async_tp)
    if validate:
        validate_tp_mesh(model, tp_mesh)
    tp_plan = get_tp_plan(model, sequence_parallel, shard_logits=shard_logits, custom_plan=custom_plan)

    tp_plan = _ensure_value_head_in_plan(model, tp_plan, sequence_parallel=sequence_parallel)
    parallelize_module(model, tp_mesh, tp_plan)

    # Re-tie weights after TP (parallelize_module may break tied embeddings)
    if ensure_tied_word_embeddings(model):
        logger.info("Re-tied embeddings after TP")

    return model
