"""
Tensor Parallelism for FSDP2
============================

This module provides DTensor tensor parallelism support including:
- Custom ParallelStyle variants (ReplicateParallel, SequenceParallelPreserveGrad)
- TP plans for common HF model families (LLaMA, Qwen, Mistral)
- Model parallelization utilities
- Ring Attention compatibility hooks
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


# === Custom ParallelStyle classes ===


class ReplicateParallel(ParallelStyle):
    """Replicate parameters as DTensors without sharding, with custom I/O layout control.

    Upstream PyTorch only provides sharding styles (Colwise/Rowwise/SequenceParallel).
    This fills the gap for modules that must NOT be sharded (e.g. score head, Q/K
    RMSNorm) but still need DTensor mesh metadata for FSDP2 grad norm & checkpointing.
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
    """SequenceParallel that preserves ``requires_grad`` when re-wrapping parameters.

    Upstream ``SequenceParallel._replicate_module_fn`` wraps params via
    ``nn.Parameter(dtensor)`` which resets requires_grad=True unconditionally,
    silently unfreezing frozen params. Critical for LoRA / partial fine-tuning.
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


# === Base utilities ===


def _parse_parallel_style(style_name: str) -> ParallelStyle:
    """Convert a string shorthand to a ``ParallelStyle`` instance.

    Args:
        style_name: One of ``"colwise"``, ``"rowwise"``, ``"colwise_rep"``,
            ``"rowwise_rep"``, ``"sequence_parallel"``, ``"replicate"``.

    Returns:
        The corresponding ``ParallelStyle`` instance.
    """
    styles: dict[str, ParallelStyle] = {
        "colwise": ColwiseParallel(),
        "rowwise": RowwiseParallel(),
        "colwise_rep": ColwiseParallel(output_layouts=Replicate()),
        "rowwise_rep": RowwiseParallel(input_layouts=Replicate()),
        "sequence_parallel": SequenceParallelPreserveGrad(),
        "replicate": ReplicateParallel(),
    }
    return styles[style_name]


# === Architecture-specific TP plan definitions ===


def _attn_mlp_plan() -> dict[str, ParallelStyle]:
    """Return TP plan entries for attention and MLP projections.

    Covers both separate (q_proj / k_proj / v_proj) and fused (qkv_proj,
    gate_up_proj) projection variants so a single plan works for models
    regardless of their linear fusion strategy.
    """
    return {
        "model.layers.*.self_attn.q_proj": ColwiseParallel(),
        "model.layers.*.self_attn.k_proj": ColwiseParallel(),
        "model.layers.*.self_attn.v_proj": ColwiseParallel(),
        "model.layers.*.self_attn.qkv_proj": ColwiseParallel(),
        "model.layers.*.self_attn.o_proj": RowwiseParallel(),
        "model.layers.*.mlp.gate_proj": ColwiseParallel(),
        "model.layers.*.mlp.up_proj": ColwiseParallel(),
        "model.layers.*.mlp.gate_up_proj": ColwiseParallel(),
        "model.layers.*.mlp.down_proj": RowwiseParallel(),
    }


def _build_default_tp_plan(
    sequence_parallel: bool = False,
    layernorm_cls: type[ParallelStyle] = SequenceParallel,
) -> dict[str, ParallelStyle]:
    """Build the default TP plan for LLaMA-style transformer models.

    This serves as the foundation plan used by all architecture-specific
    builders.  When *sequence_parallel* is True the plan adds Shard(1)
    placements for layernorms, embeddings, and residual connections.

    Args:
        sequence_parallel: Enable sequence-parallel sharding on dim 1.
        layernorm_cls: ParallelStyle class used for per-layer layernorms
            when sequence parallelism is active.

    Returns:
        A dict mapping module-name glob patterns to ``ParallelStyle`` instances.
    """
    plan: dict[str, ParallelStyle] = {
        "model.embed_tokens": RowwiseParallel(input_layouts=Replicate()),
        # Return full logits (all-gather vocab) as a local tensor for simple loss functions.
        "lm_head": ColwiseParallel(output_layouts=Replicate(), use_local_output=True),
        **_attn_mlp_plan(),
    }

    if sequence_parallel:
        plan.update(
            {
                # Embedding: output Shard(1) for SP
                "model.embed_tokens": RowwiseParallel(input_layouts=Replicate(), output_layouts=Shard(1)),
                # Final norm: SP (preserve requires_grad)
                # NOTE: return local tensor to avoid DTensor view/alias ops in HF heads
                # (e.g. `hidden_states[:, slice_indices, :]` in LlamaForCausalLM).
                "model.norm": SequenceParallelPreserveGrad(use_local_output=True),
                # LayerNorms: SP (input Shard(1), output Shard(1) as local tensor)
                "model.layers.*.input_layernorm": layernorm_cls(use_local_output=True),
                "model.layers.*.post_attention_layernorm": layernorm_cls(use_local_output=True),
                # AllGather before Attention / MLP: Shard(1) -> Replicate
                "model.layers.*.self_attn": PrepareModuleInput(
                    input_kwarg_layouts={"hidden_states": Shard(1)},
                    desired_input_kwarg_layouts={"hidden_states": Replicate()},
                    # Always feed local tensors into attention kernels (flash-attn / ring-flash-attn).
                    use_local_output=True,
                ),
                "model.layers.*.mlp": PrepareModuleInput(
                    input_layouts=Shard(1),
                    desired_input_layouts=Replicate(),
                    use_local_output=True,
                ),
                # Reduce-scatter back to Shard(1) for residual connections
                "model.layers.*.self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
                "model.layers.*.mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
                # lm_head: input Shard(1), output Replicate() (full vocab logits)
                "lm_head": ColwiseParallel(
                    input_layouts=Shard(1),
                    output_layouts=Replicate(),
                    use_local_output=True,
                ),
            }
        )

    return plan


def _build_qwen_tp_plan(model: nn.Module, sequence_parallel: bool) -> dict[str, ParallelStyle]:
    """Build TP plan for Qwen2 / Qwen3 model families.

    Extends the default plan with head-dimension sharding for the Q/K
    RMSNorm layers (``q_norm``, ``k_norm``) that operate on already
    head-sharded activations after the QKV reshape.
    """

    def _replicate_on_head_dim() -> ReplicateParallel:
        """Create a ReplicateParallel that preserves Shard(2) on the head dimension.

        Used for norms that operate on head-sharded activations with shape
        ``(batch, seq, num_heads, head_dim)`` — e.g. Q/K RMSNorm in Qwen models.
        Shard(2) marks the heads dimension so DTensor correctly all-reduces grads.
        """
        return ReplicateParallel(
            input_layout=Shard(2),
            desired_input_layout=Shard(2),
            output_layout=Shard(2),
            use_local_output=True,
        )

    plan = _build_default_tp_plan(sequence_parallel, SequenceParallelPreserveGrad)
    plan["model.layers.*.self_attn.q_norm"] = _replicate_on_head_dim()
    plan["model.layers.*.self_attn.k_norm"] = _replicate_on_head_dim()
    return plan


# Registry of architecture-specific TP plan factories.
#
# Each entry maps a HuggingFace model class name to a factory callable with signature:
#
#     def factory(model: nn.Module, sequence_parallel: bool) -> dict[str, ParallelStyle]
#
# To add support for a new model architecture:
#   1. Write a ``_build_<arch>_tp_plan`` function (or reuse an existing one).
#   2. Add the model class name and factory to this dict.
_MODEL_PLANS = {
    "Qwen2ForCausalLM": _build_qwen_tp_plan,
    "Qwen3ForCausalLM": _build_qwen_tp_plan,
}


# === Plan parsing and modification ===


def _extract_hf_tp_plan(model: nn.Module) -> dict[str, ParallelStyle] | None:
    """Extract and normalize the TP plan from a HuggingFace model's ``_tp_plan`` attribute.

    Searches the model class, instance, and inner ``model`` attribute for
    ``_tp_plan`` dicts, merges them with appropriate prefixes, and converts
    any string shorthands to ``ParallelStyle`` instances.

    Returns:
        The normalized plan dict, or ``None`` if no ``_tp_plan`` was found.
    """
    hf_entries: dict[str, ParallelStyle | str] = {}
    for src in [type(model), model, getattr(model, "model", None)]:
        if src and hasattr(src, "_tp_plan"):
            prefix = "model." if src == getattr(model, "model", None) else ""
            hf_entries.update({f"{prefix}{k}": v for k, v in src._tp_plan.items()})
    if not hf_entries:
        return None

    hf_entries.setdefault("model.embed_tokens", "rowwise_rep")
    normalized_plan: dict[str, ParallelStyle] = {
        k: _parse_parallel_style(v) if isinstance(v, str) else v for k, v in hf_entries.items()
    }

    # Always return full-vocab logits (Replicate) to keep loss functions simple.
    if "lm_head" not in normalized_plan:
        normalized_plan["lm_head"] = ColwiseParallel(output_layouts=Replicate(), use_local_output=True)
    else:
        style = normalized_plan["lm_head"]
        if isinstance(style, ColwiseParallel):
            normalized_plan["lm_head"] = ColwiseParallel(
                input_layouts=style.input_layouts[0],
                output_layouts=Replicate(),
                use_local_output=True,
            )

    return normalized_plan


def _with_loss_parallel_lm_head(plan: dict[str, ParallelStyle], sequence_parallel: bool) -> dict[str, ParallelStyle]:
    """Return a copy of *plan* with lm_head outputting vocab-sharded DTensor logits (Shard(-1)).

    This is used when loss-parallel is enabled so that the loss function
    receives sharded logits instead of the default all-gathered Replicate
    output.
    """
    plan = dict(plan)
    plan["lm_head"] = ColwiseParallel(
        input_layouts=Shard(1) if sequence_parallel else Replicate(),
        output_layouts=Shard(-1),
        use_local_output=False,
    )
    return plan


def ensure_score_layer_in_plan(
    model: nn.Module,
    plan: dict[str, ParallelStyle],
) -> dict[str, ParallelStyle]:
    """Add value-head layer handling for Reward/Critic models.

    The value head name is read from ``model.value_head_prefix`` (set by
    ``_get_reward_model`` / ``_get_critic_model``), falling back to ``"score"``.
    """
    prefix = getattr(model, "value_head_prefix", "score")
    head = getattr(model, prefix, None)
    if isinstance(head, nn.Module):
        plan = dict(plan)
        plan[prefix] = ReplicateParallel(
            input_layout=Replicate(),
            desired_input_layout=Replicate(),
            output_layout=Replicate(),
            use_local_output=True,
        )
        logger.info("Added ReplicateParallel for %s layer (Reward/Critic model)", prefix)
    return plan


def _prune_plan(plan: dict[str, ParallelStyle], model: nn.Module) -> dict[str, ParallelStyle]:
    """Drop TP plan entries that don't match any module in the model.

    Avoids noisy warnings from `parallelize_module` when a plan includes
    optional/fused module names (e.g. `qkv_proj`, `gate_up_proj`).
    """
    module_names = {name for name, _ in model.named_modules()}

    def matches(pattern: str) -> bool:
        return any(fnmatch(name, pattern) for name in module_names)

    kept = {k: v for k, v in plan.items() if matches(k)}
    if (num_pruned := len(plan) - len(kept)) > 0:
        logger.debug("Pruned %s unmatched TP plan entries", num_pruned)
    return kept


# === Public plan entry ===


def get_tp_plan(
    model: nn.Module,
    sequence_parallel: bool = False,
    custom_plan: dict[str, ParallelStyle | str] | None = None,
    shard_logits: bool = False,
) -> dict[str, ParallelStyle]:
    """Get TP plan: custom > model-specific > HuggingFace > base.

    Args:
        model: The model to parallelize.
        sequence_parallel: Enable sequence-parallel sharding.
        custom_plan: User-provided plan (string shorthands are auto-converted).
        shard_logits: If True, override lm_head to output vocab-sharded logits.

    Returns:
        A pruned dict mapping module-name patterns to ``ParallelStyle`` instances.

    Raises:
        ValueError: If no plan entries match any module in the model.
    """
    plan: dict[str, ParallelStyle] | None = None

    if custom_plan:
        plan = {k: _parse_parallel_style(v) if isinstance(v, str) else v for k, v in custom_plan.items()}
    else:
        model_cls_name = type(model).__name__
        if model_cls_name in _MODEL_PLANS:
            try:
                plan = _MODEL_PLANS[model_cls_name](model, sequence_parallel)
            except Exception as e:
                logger.warning("Plan failed for %s: %s", model_cls_name, e)

        if plan is None:
            plan = _extract_hf_tp_plan(model) or _build_default_tp_plan(sequence_parallel)

    if shard_logits:
        plan = _with_loss_parallel_lm_head(plan, sequence_parallel)
    plan = _prune_plan(plan, model)
    if not plan:
        raise ValueError(
            "No TP plan entries matched any modules for this model. "
            f"model_cls={type(model).__name__}. Provide a custom TP plan or disable TP (fsdp2_tp_size=1)."
        )
    return plan


# === Model parallelization orchestration ===


def validate_tp_mesh(model: nn.Module, tp_mesh: DeviceMesh) -> None:
    """Validate that attention heads are divisible by the TP mesh size.

    Args:
        model: The model whose config to check.
        tp_mesh: The TP device mesh.

    Raises:
        ValueError: If ``num_attention_heads`` or ``num_key_value_heads``
            is not divisible by the TP size.
    """
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
    """Enable Async Tensor Parallel (Symmetric Memory) for NVLink environments.

    See torchtitan for reference. This requires:
    1. torch.compile support
    2. NVLink / Symmetric Memory support
    """
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
    tp_plan: dict[str, ParallelStyle] | None = None,
    sequence_parallel: bool = False,
    validate: bool = True,
    enable_async_tp: bool = False,
    shard_logits: bool = False,
) -> nn.Module:
    """Apply Tensor Parallelism (DTensor parallel) to a model.

    Args:
        model: The model to parallelize.
        tp_mesh: TP device mesh. If size 1, returns the model unchanged.
        tp_plan: Pre-built TP plan. If None, one is resolved automatically.
        sequence_parallel: Enable sequence-parallel sharding.
        validate: Validate head divisibility before parallelizing.
        enable_async_tp: Enable Async TP (requires NVLink).
        shard_logits: Output vocab-sharded logits from lm_head.

    Returns:
        The parallelized model.
    """

    if tp_mesh.size() == 1:
        return model

    maybe_enable_async_tp(tp_mesh, enabled=enable_async_tp)

    if validate:
        validate_tp_mesh(model, tp_mesh)
    if tp_plan is None:
        tp_plan = get_tp_plan(model, sequence_parallel, shard_logits=shard_logits)

    tp_plan = ensure_score_layer_in_plan(model, tp_plan)

    parallelize_module(model, tp_mesh, tp_plan)

    # Re-tie weights if necessary (must happen after parallelize_module)
    if ensure_tied_word_embeddings(model):
        logger.info("Re-tied embeddings after TP")

    return model
