"""
Tensor Parallelism for FSDP2
============================

Usage:
    model = apply_tensor_parallel(model, tp_mesh, sequence_parallel=True)
    model = fully_shard(model, ...)  # Apply FSDP after TP
"""

import logging

import torch.nn as nn
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, SequenceParallel, parallelize_module

logger = logging.getLogger(__name__)


# =============================================================================
# LoRA-Aware Parallel Styles
# =============================================================================


class ColwiseParallelLora(ColwiseParallel):
    """ColwiseParallel with LoRA support - shards lora_B along dim 0."""

    src_data_rank: int = 0

    def _partition_linear_fn(self, name, module, device_mesh):
        super()._partition_linear_fn(name, module, device_mesh)
        lora_b = getattr(module, "lora_B", None) or getattr(module, "lora_b", None)
        lora_a = getattr(module, "lora_A", None) or getattr(module, "lora_a", None)
        if lora_b:
            for m in (lora_b.values() if isinstance(lora_b, nn.ModuleDict) else [lora_b]):
                _shard_param(m, "weight", device_mesh, [Shard(0)])
        if lora_a:
            for m in (lora_a.values() if isinstance(lora_a, nn.ModuleDict) else [lora_a]):
                m.register_forward_hook(
                    lambda mod, inp, out: (
                        out.redistribute(placements=[Replicate()]) if isinstance(out, DTensor) else out
                    )
                )


class RowwiseParallelLora(RowwiseParallel):
    """RowwiseParallel with LoRA support - shards lora_A along dim 1."""

    src_data_rank: int = 0

    def _partition_linear_fn(self, name, module, device_mesh):
        super()._partition_linear_fn(name, module, device_mesh)
        lora_a = getattr(module, "lora_A", None) or getattr(module, "lora_a", None)
        if lora_a:
            for m in (lora_a.values() if isinstance(lora_a, nn.ModuleDict) else [lora_a]):
                _shard_param(m, "weight", device_mesh, [Shard(1)])


class SequenceParallelAllGather(SequenceParallel):
    """SequenceParallel that all-gathers output (for LayerNorm before attention)."""

    @staticmethod
    def _prepare_output_fn(use_local_output, mod, outputs, device_mesh):
        if isinstance(outputs, DTensor) and any(isinstance(p, Shard) for p in outputs.placements):
            outputs = outputs.redistribute(device_mesh, [Replicate()])
        return SequenceParallel._prepare_output_fn(use_local_output, mod, outputs, device_mesh)


def _shard_param(module, name, mesh, placements):
    """Distribute a parameter across the mesh."""
    param = getattr(module, name, None)
    if param is not None and not isinstance(param, DTensor):
        setattr(
            module,
            name,
            nn.Parameter(distribute_tensor(param, mesh, placements, 0), requires_grad=param.requires_grad),
        )


def _to_lora(style):
    """Convert parallel style to LoRA-aware version, preserving layouts."""
    if isinstance(style, ColwiseParallel) and not isinstance(style, ColwiseParallelLora):
        new = ColwiseParallelLora()
        new.output_layouts, new.input_layouts, new.use_local_output = (
            style.output_layouts,
            style.input_layouts,
            style.use_local_output,
        )
        return new
    if isinstance(style, RowwiseParallel) and not isinstance(style, RowwiseParallelLora):
        new = RowwiseParallelLora()
        new.output_layouts, new.input_layouts, new.use_local_output = (
            style.output_layouts,
            style.input_layouts,
            style.use_local_output,
        )
        return new
    return style


# =============================================================================
# TP Plans
# =============================================================================

# Attention + MLP layers (shared by all LLaMA-style models)
_ATTN_MLP_PLAN = {
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


def _base_plan(sequence_parallel=False, layernorm_cls=SequenceParallel):
    """Base TP plan for LLaMA-style models."""
    plan = {
        "model.embed_tokens": RowwiseParallel(input_layouts=Replicate()),
        "lm_head": ColwiseParallel(output_layouts=Shard(-1), use_local_output=False),
        **_ATTN_MLP_PLAN,
    }
    if sequence_parallel:
        plan.update(
            {
                "model.embed_tokens": RowwiseParallel(input_layouts=Replicate(), output_layouts=Shard(1)),
                "model.norm": SequenceParallel(),
                "model.layers.*.input_layernorm": layernorm_cls(use_local_output=False),
                "model.layers.*.post_attention_layernorm": layernorm_cls(use_local_output=False),
                "model.layers.*.self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
                "model.layers.*.mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
                "lm_head": ColwiseParallel(input_layouts=Shard(1), output_layouts=Shard(-1), use_local_output=False),
            }
        )
    return plan


def _llama_plan(model, sequence_parallel):
    """LLaMA/Mistral plan with SequenceParallelAllGather for LayerNorm."""
    return _base_plan(sequence_parallel, SequenceParallelAllGather)


def _qwen_plan(model, sequence_parallel):
    """Qwen plan: adds q_norm/k_norm handling."""
    plan = _base_plan(sequence_parallel, SequenceParallelAllGather)
    if sequence_parallel:

        class QwenQKNorm(SequenceParallel):
            @staticmethod
            def _prepare_input_fn(seq_shard, mod, inputs, mesh):
                t = inputs[0]
                return t if isinstance(t, DTensor) else DTensor.from_local(t, mesh, seq_shard, run_check=False)

        plan["model.layers.*.self_attn.q_norm"] = QwenQKNorm()
        plan["model.layers.*.self_attn.k_norm"] = QwenQKNorm()
    return plan


# Model class -> plan function
_MODEL_PLANS = {
    "LlamaForCausalLM": _llama_plan,
    "LlamaForSequenceClassification": _llama_plan,
    "MistralForCausalLM": _llama_plan,
    "Qwen2ForCausalLM": _qwen_plan,
    "Qwen3ForCausalLM": _qwen_plan,
}


def get_tp_plan(model, sequence_parallel=False, custom_plan=None):
    """Get TP plan: custom > model-specific > HuggingFace > base."""
    if custom_plan:
        return {k: _str_to_style(v) if isinstance(v, str) else v for k, v in custom_plan.items()}

    cls_name = type(model).__name__
    if cls_name in _MODEL_PLANS:
        try:
            return _MODEL_PLANS[cls_name](model, sequence_parallel)
        except Exception as e:
            logger.warning(f"Plan failed for {cls_name}: {e}")

    hf_plan = _get_hf_plan(model)
    if hf_plan:
        return hf_plan

    return _base_plan(sequence_parallel)


def _str_to_style(s):
    """Convert string to ParallelStyle."""
    styles = {
        "colwise": ColwiseParallel(),
        "rowwise": RowwiseParallel(),
        "colwise_rep": ColwiseParallel(output_layouts=Replicate()),
        "rowwise_rep": RowwiseParallel(input_layouts=Replicate()),
        "sequence_parallel": SequenceParallel(),
    }
    return styles[s]


def _get_hf_plan(model):
    """Extract TP plan from HuggingFace model's _tp_plan attribute."""
    hf = {}
    for src in [type(model), model, getattr(model, "model", None)]:
        if src and hasattr(src, "_tp_plan"):
            prefix = "model." if src == getattr(model, "model", None) else ""
            hf.update({f"{prefix}{k}": v for k, v in src._tp_plan.items()})
    if not hf:
        return None

    hf.setdefault("model.embed_tokens", "rowwise_rep")
    result = {k: _str_to_style(v) if isinstance(v, str) else v for k, v in hf.items()}
    if "lm_head" in result and isinstance(result["lm_head"], ColwiseParallel):
        result["lm_head"] = ColwiseParallel(output_layouts=Shard(-1), use_local_output=False)
    return result


# =============================================================================
# Application
# =============================================================================


def validate_tp_mesh(model, tp_mesh):
    """Validate attention heads divisible by TP size."""
    if tp_mesh.size() == 1:
        return
    cfg = getattr(model, "config", None)
    if not cfg:
        return
    tp = tp_mesh.size()
    heads = getattr(cfg, "num_attention_heads", 0)
    kv = getattr(cfg, "num_key_value_heads", 0) or heads
    if heads and heads % tp:
        raise ValueError(f"num_attention_heads ({heads}) not divisible by TP ({tp})")
    if kv and kv % tp:
        raise ValueError(f"num_key_value_heads ({kv}) not divisible by TP ({tp})")


def apply_tensor_parallel(model, tp_mesh, tp_plan=None, sequence_parallel=False, validate=True):
    """Apply Tensor Parallelism to model."""
    if tp_mesh.size() == 1:
        return model
    if validate:
        validate_tp_mesh(model, tp_mesh)
    if tp_plan is None:
        tp_plan = get_tp_plan(model, sequence_parallel)

    # Convert to LoRA-aware styles and rely on parallelize_module's fnmatch-based
    # wildcard matching (e.g., 'model.layers.*.self_attn.q_proj').
    tp_plan = {k: _to_lora(v) for k, v in tp_plan.items()}
    parallelize_module(model, tp_mesh, tp_plan)
    return model
