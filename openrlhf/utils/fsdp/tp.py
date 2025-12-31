"""
Tensor Parallelism for FSDP2
============================

Provides:
- LoRA-aware parallel styles (ColwiseParallelLora, RowwiseParallelLora)
- TP plan generation (model-specific > HuggingFace > generic)
- apply_tensor_parallel(): main entry point

Usage:
    model = apply_tensor_parallel(model, tp_mesh, sequence_parallel=True)
    model = fully_shard(model, ...)  # Apply FSDP after TP
"""

import logging
import re
from functools import lru_cache
from typing import Callable, Dict, Optional, Union

import torch.nn as nn
from torch import Tensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from torch.distributed.tensor.parallel import (
    ColwiseParallel, RowwiseParallel, SequenceParallel, ParallelStyle, parallelize_module
)

logger = logging.getLogger(__name__)


# =============================================================================
# LoRA-Aware Parallel Styles
# =============================================================================
# When using LoRA with TP, we need to shard LoRA weights correctly:
# - ColwiseParallel (q/k/v/gate/up): shard lora_B along dim 0
# - RowwiseParallel (o/down): shard lora_A along dim 1

class ColwiseParallelLora(ColwiseParallel):
    """ColwiseParallel with LoRA support - shards lora_B weight."""
    src_data_rank: int = 0

    def _partition_linear_fn(self, name: str, module: nn.Module, device_mesh: DeviceMesh):
        super()._partition_linear_fn(name, module, device_mesh)
        # Shard lora_B (output dimension)
        lora_b = getattr(module, "lora_B", None) or getattr(module, "lora_b", None)
        if lora_b:
            for m in (lora_b.values() if isinstance(lora_b, nn.ModuleDict) else [lora_b]):
                _shard_param(m, "weight", device_mesh, [Shard(0)], self.src_data_rank)
        # Add all-gather hook to lora_A output
        lora_a = getattr(module, "lora_A", None) or getattr(module, "lora_a", None)
        if lora_a:
            for m in (lora_a.values() if isinstance(lora_a, nn.ModuleDict) else [lora_a]):
                _add_allgather_hook(m)


class RowwiseParallelLora(RowwiseParallel):
    """RowwiseParallel with LoRA support - shards lora_A weight."""
    src_data_rank: int = 0

    def _partition_linear_fn(self, name: str, module: nn.Module, device_mesh: DeviceMesh):
        super()._partition_linear_fn(name, module, device_mesh)
        # Shard lora_A (input dimension)
        lora_a = getattr(module, "lora_A", None) or getattr(module, "lora_a", None)
        if lora_a:
            for m in (lora_a.values() if isinstance(lora_a, nn.ModuleDict) else [lora_a]):
                _shard_param(m, "weight", device_mesh, [Shard(1)], self.src_data_rank)


class SequenceParallelAllGather(SequenceParallel):
    """SequenceParallel that all-gathers output (for LayerNorm before attention)."""
    @staticmethod
    def _prepare_output_fn(use_local_output, mod, outputs, device_mesh):
        if isinstance(outputs, DTensor) and any(isinstance(p, Shard) for p in outputs.placements):
            outputs = outputs.redistribute(device_mesh, [Replicate()])
        return SequenceParallel._prepare_output_fn(use_local_output, mod, outputs, device_mesh)


def _shard_param(module, name, mesh, placements, src=0):
    """Distribute a parameter across the mesh."""
    param = getattr(module, name, None)
    if param is None or isinstance(param, DTensor):
        return
    setattr(module, name, nn.Parameter(
        distribute_tensor(param, mesh, placements, src), requires_grad=param.requires_grad))


def _add_allgather_hook(module):
    """Add forward hook to all-gather DTensor output."""
    def hook(mod, inp, out: Tensor) -> Tensor:
        return out.redistribute(placements=[Replicate()]) if isinstance(out, DTensor) else out
    module.register_forward_hook(hook)


def translate_to_lora(style: ParallelStyle) -> ParallelStyle:
    """Convert parallel style to LoRA-aware version."""
    if isinstance(style, ColwiseParallel) and not isinstance(style, ColwiseParallelLora):
        new = ColwiseParallelLora()
        new.output_layouts, new.input_layouts, new.use_local_output = style.output_layouts, style.input_layouts, style.use_local_output
        return new
    if isinstance(style, RowwiseParallel) and not isinstance(style, RowwiseParallelLora):
        new = RowwiseParallelLora()
        new.output_layouts, new.input_layouts, new.use_local_output = style.output_layouts, style.input_layouts, style.use_local_output
        return new
    return style


# =============================================================================
# TP Plans
# =============================================================================
# TP plans specify which layers get which parallel style.
# Wildcards like "model.layers.*.self_attn.q_proj" are expanded to match all layers.

@lru_cache
def _str_to_style(s: str) -> ParallelStyle:
    """Convert string to ParallelStyle."""
    return {
        "colwise": ColwiseParallel(),
        "rowwise": RowwiseParallel(),
        "colwise_rep": ColwiseParallel(output_layouts=Replicate()),
        "rowwise_rep": RowwiseParallel(input_layouts=Replicate()),
        "sequence_parallel": SequenceParallel(),
    }[s]


def _attn_mlp_plan() -> Dict[str, ParallelStyle]:
    """Common plan for attention + MLP layers."""
    return {
        # Attention: q/k/v are colwise, o is rowwise
        "model.layers.*.self_attn.q_proj": ColwiseParallel(),
        "model.layers.*.self_attn.k_proj": ColwiseParallel(),
        "model.layers.*.self_attn.v_proj": ColwiseParallel(),
        "model.layers.*.self_attn.qkv_proj": ColwiseParallel(),  # fused
        "model.layers.*.self_attn.o_proj": RowwiseParallel(),
        # MLP: gate/up are colwise, down is rowwise
        "model.layers.*.mlp.gate_proj": ColwiseParallel(),
        "model.layers.*.mlp.up_proj": ColwiseParallel(),
        "model.layers.*.mlp.gate_up_proj": ColwiseParallel(),    # fused
        "model.layers.*.mlp.down_proj": RowwiseParallel(),
    }


def get_base_tp_plan(sequence_parallel: bool = False) -> Dict[str, ParallelStyle]:
    """Generic TP plan for LLaMA-style models."""
    plan = {
        "model.embed_tokens": RowwiseParallel(input_layouts=Replicate()),
        "lm_head": ColwiseParallel(output_layouts=Shard(-1), use_local_output=False),
        **_attn_mlp_plan(),
    }
    if sequence_parallel:
        # With SP: shard activations along sequence dimension between layers
        plan.update({
            "model.embed_tokens": RowwiseParallel(input_layouts=Replicate(), output_layouts=Shard(1)),
            "model.norm": SequenceParallel(),
            "model.layers.*.input_layernorm": SequenceParallel(),
            "model.layers.*.post_attention_layernorm": SequenceParallel(),
            "model.layers.*.self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
            "model.layers.*.mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
            "lm_head": ColwiseParallel(input_layouts=Shard(1), output_layouts=Replicate()),
        })
    return plan


def _llama_plan(model, sp: bool):
    """LLaMA/Mistral plan with SequenceParallelAllGather."""
    plan = get_base_tp_plan(False)
    if sp:
        plan.update({
            "model.embed_tokens": RowwiseParallel(input_layouts=Replicate(), output_layouts=Shard(1)),
            "model.norm": SequenceParallel(),
            "model.layers.*.input_layernorm": SequenceParallelAllGather(use_local_output=False),
            "model.layers.*.post_attention_layernorm": SequenceParallelAllGather(use_local_output=False),
            "model.layers.*.self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
            "model.layers.*.mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
            "lm_head": ColwiseParallel(input_layouts=Shard(1), output_layouts=Shard(-1), use_local_output=False),
        })
    return plan


def _qwen_plan(model, sp: bool):
    """Qwen plan with q_norm/k_norm handling."""
    plan = {"model.embed_tokens": RowwiseParallel(input_layouts=Replicate()),
            "lm_head": ColwiseParallel(output_layouts=Shard(-1), use_local_output=False),
            **_attn_mlp_plan()}
    if sp:
        class QwenQKNorm(SequenceParallel):
            @staticmethod
            def _prepare_input_fn(seq_shard, mod, inputs, mesh):
                t = inputs[0]
                return t if isinstance(t, DTensor) else DTensor.from_local(t, mesh, seq_shard, run_check=False)

        plan.update({
            "model.embed_tokens": RowwiseParallel(input_layouts=Replicate(), output_layouts=Shard(1)),
            "model.norm": SequenceParallel(),
            "model.layers.*.input_layernorm": SequenceParallelAllGather(),
            "model.layers.*.post_attention_layernorm": SequenceParallelAllGather(),
            "model.layers.*.self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
            "model.layers.*.self_attn.q_norm": QwenQKNorm(),
            "model.layers.*.self_attn.k_norm": QwenQKNorm(),
            "model.layers.*.mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
            "lm_head": ColwiseParallel(input_layouts=Shard(1), output_layouts=Shard(-1), use_local_output=False),
        })
    return plan


# Model class name -> plan function
_MODEL_PLANS = {
    "LlamaForCausalLM": _llama_plan,
    "LlamaForSequenceClassification": _llama_plan,
    "MistralForCausalLM": _llama_plan,
    "Qwen2ForCausalLM": _qwen_plan,
    "Qwen3ForCausalLM": _qwen_plan,
}


def get_tp_plan(model: nn.Module, sequence_parallel: bool = False,
                custom_plan: Optional[Dict[str, Union[str, ParallelStyle]]] = None) -> Dict[str, ParallelStyle]:
    """Get TP plan: custom > model-specific > HuggingFace > generic."""
    # 1. Custom plan
    if custom_plan:
        return {k: _str_to_style(v) if isinstance(v, str) else v for k, v in custom_plan.items()}

    # 2. Model-specific plan
    cls_name = type(model).__name__
    if cls_name in _MODEL_PLANS:
        try:
            return _MODEL_PLANS[cls_name](model, sequence_parallel)
        except Exception as e:
            logger.warning(f"Model-specific plan failed for {cls_name}: {e}")

    # 3. HuggingFace _tp_plan attribute
    hf_plan = _get_hf_plan(model)
    if hf_plan:
        return hf_plan

    # 4. Generic base plan
    return get_base_tp_plan(sequence_parallel)


def _get_hf_plan(model) -> Optional[Dict[str, ParallelStyle]]:
    """Extract TP plan from HuggingFace model's _tp_plan attribute."""
    hf = {}
    for src in [type(model), model, getattr(model, "model", None)]:
        if src and hasattr(src, "_tp_plan"):
            prefix = "model." if src == getattr(model, "model", None) else ""
            for k, v in src._tp_plan.items():
                hf[f"{prefix}{k}"] = v
    if not hf:
        return None

    # Ensure embed_tokens is included
    if "model.embed_tokens" not in hf:
        hf["model.embed_tokens"] = "rowwise_rep"

    # Convert strings to styles and fix lm_head
    result = {k: _str_to_style(v) if isinstance(v, str) else v for k, v in hf.items()}
    if "lm_head" in result and isinstance(result["lm_head"], ColwiseParallel):
        result["lm_head"] = ColwiseParallel(output_layouts=Shard(-1), use_local_output=False)
    return result


# =============================================================================
# Application
# =============================================================================

def validate_tp_mesh(model: nn.Module, tp_mesh: DeviceMesh):
    """Validate attention heads are divisible by TP size."""
    if tp_mesh.size() == 1:
        return
    cfg = getattr(model, "config", None)
    if not cfg:
        return
    tp = tp_mesh.size()
    heads = getattr(cfg, "num_attention_heads", 0)
    kv = getattr(cfg, "num_key_value_heads", heads)
    if heads and heads % tp:
        raise ValueError(f"num_attention_heads ({heads}) not divisible by TP size ({tp})")
    if kv and kv % tp:
        raise ValueError(f"num_key_value_heads ({kv}) not divisible by TP size ({tp})")


def _expand_wildcards(plan: Dict[str, ParallelStyle], model: nn.Module) -> Dict[str, ParallelStyle]:
    """Expand wildcard patterns like 'model.layers.*.proj' to actual module names."""
    names = {n for n, _ in model.named_modules()}
    result = {}
    for pattern, style in plan.items():
        if "*" not in pattern:
            if pattern in names:
                result[pattern] = style
        else:
            # Convert pattern to regex: model.layers.* -> model.layers.\d+
            regex = re.compile(f"^{pattern.replace('.', r'\.').replace('*', r'\\d+')}$")
            for n in names:
                if regex.match(n):
                    result[n] = style
    return result


def apply_tensor_parallel(model: nn.Module, tp_mesh: DeviceMesh, tp_plan: Optional[Dict] = None,
                          sequence_parallel: bool = False, validate: bool = True) -> nn.Module:
    """Apply Tensor Parallelism to model.
    
    Args:
        model: Model to parallelize
        tp_mesh: DeviceMesh for TP dimension
        tp_plan: Custom TP plan (optional, auto-generated if None)
        sequence_parallel: Enable sequence parallelism
        validate: Validate head count divisibility
    
    Returns:
        Parallelized model (in-place modification)
    """
    if tp_mesh.size() == 1:
        return model

    if validate:
        validate_tp_mesh(model, tp_mesh)

    if tp_plan is None:
        tp_plan = get_tp_plan(model, sequence_parallel)

    # Convert to LoRA-aware styles
    tp_plan = {k: translate_to_lora(v) for k, v in tp_plan.items()}

    # Expand wildcards and apply
    expanded = _expand_wildcards(tp_plan, model)
    if expanded:
        parallelize_module(model, tp_mesh, expanded)

    return model


def register_tp_plan(model_cls_name: str, plan_fn: Callable):
    """Register custom TP plan for a model class."""
    _MODEL_PLANS[model_cls_name] = plan_fn
