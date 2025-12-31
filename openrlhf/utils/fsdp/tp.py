"""
Tensor Parallelism for FSDP2.

Provides TP plan generation, LoRA-aware parallel styles, and model parallelization.
Based on NeMo Automodel and PyTorch native API.

Usage:
    model = apply_tensor_parallel(model, tp_mesh)
    model = fully_shard(model, ...)
"""

import logging
import re
from functools import lru_cache
from typing import Callable, Dict, Optional, Sequence, Union

import torch.nn as nn
from torch import Tensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    ParallelStyle,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)
from torch.distributed.tensor.placement_types import Placement

logger = logging.getLogger(__name__)


# =============================================================================
# LoRA-Aware Parallel Styles
# =============================================================================


class ColwiseParallelLora(ColwiseParallel):
    """ColwiseParallel with LoRA support."""

    src_data_rank: int = 0

    def _partition_linear_fn(self, name: str, module: nn.Module, device_mesh: DeviceMesh) -> None:
        super()._partition_linear_fn(name, module, device_mesh)
        lora_a = getattr(module, "lora_A", None) or getattr(module, "lora_a", None)
        lora_b = getattr(module, "lora_B", None) or getattr(module, "lora_b", None)

        if lora_b:
            targets = lora_b.values() if isinstance(lora_b, nn.ModuleDict) else [lora_b]
            for t in targets:
                _shard_param(t, "weight", device_mesh, [Shard(0)], self.src_data_rank)

        if lora_a:
            targets = lora_a.values() if isinstance(lora_a, nn.ModuleDict) else [lora_a]
            for t in targets:
                _add_allgather_hook(t)


class RowwiseParallelLora(RowwiseParallel):
    """RowwiseParallel with LoRA support."""

    src_data_rank: int = 0

    def _partition_linear_fn(self, name: str, module: nn.Module, device_mesh: DeviceMesh) -> None:
        super()._partition_linear_fn(name, module, device_mesh)
        lora_a = getattr(module, "lora_A", None) or getattr(module, "lora_a", None)

        if lora_a:
            targets = lora_a.values() if isinstance(lora_a, nn.ModuleDict) else [lora_a]
            for t in targets:
                _shard_param(t, "weight", device_mesh, [Shard(1)], self.src_data_rank)


class SequenceParallelLora(SequenceParallel):
    """SequenceParallel with LoRA support."""

    src_data_rank: int = 0


class SequenceParallelAllGather(SequenceParallel):
    """SequenceParallel that all-gathers output."""

    @staticmethod
    def _prepare_output_fn(use_local_output, mod, outputs, device_mesh):
        if isinstance(outputs, DTensor) and any(isinstance(p, Shard) for p in outputs.placements):
            outputs = outputs.redistribute(device_mesh, [Replicate()])
        return SequenceParallel._prepare_output_fn(use_local_output, mod, outputs, device_mesh)


def _shard_param(module: nn.Module, name: str, mesh: DeviceMesh, placements: Sequence[Placement], src: int = 0):
    param = getattr(module, name, None)
    if param is None or isinstance(param, DTensor):
        return
    setattr(
        module, name, nn.Parameter(distribute_tensor(param, mesh, placements, src), requires_grad=param.requires_grad)
    )


def _add_allgather_hook(module: nn.Module):
    def hook(mod, inp, out: Tensor) -> Tensor:
        return out.redistribute(placements=[Replicate()]) if isinstance(out, DTensor) else out

    module.register_forward_hook(hook)


def translate_to_lora(style):
    """Convert parallel style to LoRA-aware version."""
    if isinstance(style, ColwiseParallel) and not isinstance(style, ColwiseParallelLora):
        s = ColwiseParallelLora()
        s.output_layouts, s.input_layouts, s.use_local_output = (
            style.output_layouts,
            style.input_layouts,
            style.use_local_output,
        )
        return s
    if isinstance(style, RowwiseParallel) and not isinstance(style, RowwiseParallelLora):
        s = RowwiseParallelLora()
        s.output_layouts, s.input_layouts, s.use_local_output = (
            style.output_layouts,
            style.input_layouts,
            style.use_local_output,
        )
        return s
    if isinstance(style, SequenceParallel) and not isinstance(style, SequenceParallelLora):
        return SequenceParallelLora()
    return style


# =============================================================================
# TP Plans
# =============================================================================


@lru_cache
def _str_to_style(style: str) -> ParallelStyle:
    styles = {
        "colwise": ColwiseParallel(),
        "rowwise": RowwiseParallel(),
        "colwise_rep": ColwiseParallel(output_layouts=Replicate()),
        "rowwise_rep": RowwiseParallel(input_layouts=Replicate()),
        "sequence_parallel": SequenceParallel(),
    }
    if style not in styles:
        raise ValueError(f"Unknown style: {style}")
    return styles[style]


def get_base_tp_plan(sequence_parallel: bool = False) -> Dict[str, ParallelStyle]:
    """Generic TP plan for LLaMA-style models."""
    plan = {
        "model.embed_tokens": RowwiseParallel(input_layouts=Replicate()),
        "model.layers.*.self_attn.q_proj": ColwiseParallel(),
        "model.layers.*.self_attn.k_proj": ColwiseParallel(),
        "model.layers.*.self_attn.v_proj": ColwiseParallel(),
        "model.layers.*.self_attn.qkv_proj": ColwiseParallel(),
        "model.layers.*.self_attn.o_proj": RowwiseParallel(),
        "model.layers.*.mlp.gate_proj": ColwiseParallel(),
        "model.layers.*.mlp.up_proj": ColwiseParallel(),
        "model.layers.*.mlp.gate_up_proj": ColwiseParallel(),
        "model.layers.*.mlp.down_proj": RowwiseParallel(),
        "lm_head": ColwiseParallel(output_layouts=Shard(-1), use_local_output=False),
    }
    if sequence_parallel:
        plan.update(
            {
                "model.embed_tokens": RowwiseParallel(input_layouts=Replicate(), output_layouts=Shard(1)),
                "model.norm": SequenceParallel(),
                "model.layers.*.input_layernorm": SequenceParallel(),
                "model.layers.*.self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
                "model.layers.*.post_attention_layernorm": SequenceParallel(),
                "model.layers.*.mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
                "lm_head": ColwiseParallel(input_layouts=Shard(1), output_layouts=Replicate()),
            }
        )
    return plan


def _llama_plan(model, sp: bool) -> Dict[str, ParallelStyle]:
    """Optimized plan for LLaMA."""
    plan = get_base_tp_plan(False)
    if sp:
        plan.update(
            {
                "model.embed_tokens": RowwiseParallel(input_layouts=Replicate(), output_layouts=Shard(1)),
                "model.norm": SequenceParallel(),
                "model.layers.*.input_layernorm": SequenceParallelAllGather(use_local_output=False),
                "model.layers.*.post_attention_layernorm": SequenceParallelAllGather(use_local_output=False),
                "model.layers.*.self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
                "model.layers.*.mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
                "lm_head": ColwiseParallel(input_layouts=Shard(1), output_layouts=Shard(-1), use_local_output=False),
            }
        )
    return plan


def _qwen_plan(model, sp: bool) -> Dict[str, ParallelStyle]:
    """Optimized plan for Qwen."""
    if sp:

        class Qwen3QKNorm(SequenceParallel):
            @staticmethod
            def _prepare_input_fn(seq_shard, mod, inputs, mesh):
                t = inputs[0]
                if isinstance(t, DTensor):
                    return t
                return DTensor.from_local(t, mesh, seq_shard, run_check=False)

        return {
            "lm_head": ColwiseParallel(input_layouts=Shard(1), output_layouts=Shard(-1), use_local_output=False),
            "model.embed_tokens": RowwiseParallel(input_layouts=Replicate(), output_layouts=Shard(1)),
            "model.norm": SequenceParallel(),
            "model.layers.*.input_layernorm": SequenceParallelAllGather(),
            "model.layers.*.self_attn.q_proj": ColwiseParallel(),
            "model.layers.*.self_attn.k_proj": ColwiseParallel(),
            "model.layers.*.self_attn.v_proj": ColwiseParallel(),
            "model.layers.*.self_attn.qkv_proj": ColwiseParallel(),
            "model.layers.*.self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
            "model.layers.*.self_attn.q_norm": Qwen3QKNorm(),
            "model.layers.*.self_attn.k_norm": Qwen3QKNorm(),
            "model.layers.*.post_attention_layernorm": SequenceParallelAllGather(),
            "model.layers.*.mlp.up_proj": ColwiseParallel(),
            "model.layers.*.mlp.gate_proj": ColwiseParallel(),
            "model.layers.*.mlp.gate_up_proj": ColwiseParallel(),
            "model.layers.*.mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
        }
    return {
        "lm_head": ColwiseParallel(output_layouts=Shard(-1), use_local_output=False),
        "model.embed_tokens": RowwiseParallel(input_layouts=Replicate()),
        "model.layers.*.self_attn.q_proj": ColwiseParallel(),
        "model.layers.*.self_attn.k_proj": ColwiseParallel(),
        "model.layers.*.self_attn.v_proj": ColwiseParallel(),
        "model.layers.*.self_attn.qkv_proj": ColwiseParallel(),
        "model.layers.*.self_attn.o_proj": RowwiseParallel(),
        "model.layers.*.mlp.up_proj": ColwiseParallel(),
        "model.layers.*.mlp.gate_proj": ColwiseParallel(),
        "model.layers.*.mlp.gate_up_proj": ColwiseParallel(),
        "model.layers.*.mlp.down_proj": RowwiseParallel(),
    }


_MODEL_PLANS: Dict[str, Callable] = {
    "LlamaForCausalLM": _llama_plan,
    "LlamaForSequenceClassification": _llama_plan,
    "Qwen2ForCausalLM": _qwen_plan,
    "Qwen3ForCausalLM": _qwen_plan,
    "MistralForCausalLM": _llama_plan,
}


def get_tp_plan(
    model: nn.Module,
    sequence_parallel: bool = False,
    custom_plan: Optional[Dict[str, Union[str, ParallelStyle]]] = None,
) -> Dict[str, ParallelStyle]:
    """Get TP plan: custom > model-specific > HF > base."""
    if custom_plan:
        return {k: _str_to_style(v) if isinstance(v, str) else v for k, v in custom_plan.items()}

    # Model-specific
    cls_name = type(model).__name__
    if cls_name in _MODEL_PLANS:
        try:
            return _MODEL_PLANS[cls_name](model, sequence_parallel)
        except Exception as e:
            logger.warning(f"Failed to get plan for {cls_name}: {e}")

    # HF plan
    hf = {}
    for src in [type(model), model, getattr(model, "model", None)]:
        if src and hasattr(src, "_tp_plan"):
            prefix = "model." if src == getattr(model, "model", None) else ""
            for k, v in src._tp_plan.items():
                hf[f"{prefix}{k}"] = v
    if hf:
        if "model.embed_tokens" not in hf:
            hf["model.embed_tokens"] = "rowwise_rep"
        result = {}
        for k, v in hf.items():
            result[k] = _str_to_style(v) if isinstance(v, str) else v
        if "lm_head" in result and isinstance(result["lm_head"], ColwiseParallel):
            result["lm_head"] = ColwiseParallel(output_layouts=Shard(-1), use_local_output=False)
        return result

    return get_base_tp_plan(sequence_parallel)


# =============================================================================
# Application
# =============================================================================


def validate_tp_mesh(model: nn.Module, tp_mesh: DeviceMesh) -> None:
    """Validate attention heads divisible by TP size."""
    if tp_mesh.size() == 1:
        return
    cfg = getattr(model, "config", None)
    if not cfg:
        return
    tp = tp_mesh.size()
    heads = getattr(cfg, "num_attention_heads", 0)
    kv = getattr(cfg, "num_key_value_heads", heads)
    if heads and heads % tp:
        raise ValueError(f"num_attention_heads ({heads}) not divisible by TP ({tp})")
    if kv and kv % tp:
        raise ValueError(f"num_key_value_heads ({kv}) not divisible by TP ({tp})")


def _expand_wildcards(plan: Dict[str, ParallelStyle], model: nn.Module) -> Dict[str, ParallelStyle]:
    names = {n for n, _ in model.named_modules()}
    result = {}
    for pattern, style in plan.items():
        if "*" not in pattern:
            if pattern in names or pattern == "":
                result[pattern] = style
        else:
            regex = re.compile(f"^{pattern.replace('.', r'\.').replace('*', r'\d+')}$")
            for n in names:
                if regex.match(n):
                    result[n] = style
    return result


def apply_tensor_parallel(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    tp_plan: Optional[Dict[str, ParallelStyle]] = None,
    sequence_parallel: bool = False,
    validate: bool = True,
) -> nn.Module:
    """Apply Tensor Parallelism to model."""
    if tp_mesh.size() == 1:
        return model

    if validate:
        validate_tp_mesh(model, tp_mesh)

    if tp_plan is None:
        tp_plan = get_tp_plan(model, sequence_parallel)

    tp_plan = {k: translate_to_lora(v) for k, v in tp_plan.items()}

    if tp_plan:
        expanded = _expand_wildcards(tp_plan, model)
        if expanded:
            parallelize_module(model, tp_mesh, expanded)

    return model


def register_tp_plan(model_cls_name: str, plan_fn: Callable) -> None:
    """Register custom TP plan for a model class."""
    _MODEL_PLANS[model_cls_name] = plan_fn
