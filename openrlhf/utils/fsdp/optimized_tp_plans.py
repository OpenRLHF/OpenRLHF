"""
Model-specific optimized TP Plans.

Based on NeMo Automodel implementation, provides optimized Tensor Parallel plans
for different model architectures with better performance and compatibility
compared to the generic base plan.

Supported models:
- LLaMA series (LlamaForCausalLM)
- Qwen series (Qwen2ForCausalLM, Qwen3ForCausalLM)
- Gemma series (to be added)
"""

import logging
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _is_torch_tp_available() -> bool:
    """Check if PyTorch version supports tensor parallel API."""
    try:
        from packaging import version

        torch_version = version.parse(torch.__version__.split("+")[0])
        return torch_version >= version.parse("2.5.0")
    except Exception:
        return False


if _is_torch_tp_available():
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor.parallel import (
        ColwiseParallel,
        ParallelStyle,
        RowwiseParallel,
        SequenceParallel,
    )
    from torch.distributed.tensor.placement_types import Replicate, Shard


class SequenceParallelAllGatherActivation(SequenceParallel):
    """
    SequenceParallel with all-gather on activations.

    Performs all-gather on output to ensure activations are complete after LayerNorm.
    Used in conjunction with RowwiseParallel attention/mlp layers.
    """

    @staticmethod
    def _prepare_output_fn(use_local_output, mod, outputs, device_mesh):
        """Redistribute output to Replicate."""
        if isinstance(outputs, DTensor):
            if any(isinstance(p, Shard) for p in outputs.placements):
                outputs = outputs.redistribute(
                    device_mesh=device_mesh,
                    placements=[Replicate()],
                )
        return SequenceParallel._prepare_output_fn(use_local_output, mod, outputs, device_mesh)


def _parallelize_llama(
    model: nn.Module,
    sequence_parallel: bool = False,
) -> Dict[str, "ParallelStyle"]:
    """
    Generate optimized TP plan for LLaMA series models.

    LLaMA architecture characteristics:
    - Separate Q/K/V projections
    - Optional fused QKV (qkv_proj)
    - GatedMLP with separate gate/up (gate_proj, up_proj) or fused (gate_up_proj)
    - RMSNorm

    Args:
        model: LLaMA model
        sequence_parallel: Whether to enable Sequence Parallel

    Returns:
        TP plan dictionary
    """
    if not _is_torch_tp_available():
        raise RuntimeError("Tensor parallel requires PyTorch >= 2.5.0")

    base_plan: Dict[str, ParallelStyle] = {
        # Embedding: shard along embedding_dim, input Replicate
        "model.embed_tokens": RowwiseParallel(input_layouts=Replicate()),
        # Attention projections: Colwise sharding
        "model.layers.*.self_attn.q_proj": ColwiseParallel(),
        "model.layers.*.self_attn.k_proj": ColwiseParallel(),
        "model.layers.*.self_attn.v_proj": ColwiseParallel(),
        "model.layers.*.self_attn.qkv_proj": ColwiseParallel(),  # Fused QKV
        "model.layers.*.self_attn.o_proj": RowwiseParallel(),
        # MLP projections
        "model.layers.*.mlp.gate_proj": ColwiseParallel(),
        "model.layers.*.mlp.up_proj": ColwiseParallel(),
        "model.layers.*.mlp.gate_up_proj": ColwiseParallel(),  # Fused gate+up
        "model.layers.*.mlp.down_proj": RowwiseParallel(),
        # lm_head: Colwise sharding, keep output Shard to reduce communication
        "lm_head": ColwiseParallel(output_layouts=Shard(-1), use_local_output=False),
    }

    if sequence_parallel:
        sp_plan = {
            # Embedding output changed to Shard(1) to initiate SP
            "model.embed_tokens": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            # LayerNorm uses SP (input Shard(1), output Shard(1))
            "model.norm": SequenceParallel(),
            "model.layers.*.input_layernorm": SequenceParallelAllGatherActivation(use_local_output=False),
            "model.layers.*.post_attention_layernorm": SequenceParallelAllGatherActivation(use_local_output=False),
            # Rowwise layers output changed to Shard(1)
            "model.layers.*.self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
            "model.layers.*.mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
            # lm_head input changed to Shard(1)
            "lm_head": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Shard(-1),
                use_local_output=False,
            ),
        }
        base_plan.update(sp_plan)

    return base_plan


def _parallelize_qwen(
    model: nn.Module,
    sequence_parallel: bool = False,
) -> Dict[str, "ParallelStyle"]:
    """
    Generate optimized TP plan for Qwen2/Qwen3 series models.

    Qwen architecture characteristics:
    - Similar to LLaMA structure
    - Qwen3 has q_norm and k_norm (QK LayerNorm)
    - May have fused QKV (qkv_proj)

    Args:
        model: Qwen model
        sequence_parallel: Whether to enable Sequence Parallel

    Returns:
        TP plan dictionary
    """
    if not _is_torch_tp_available():
        raise RuntimeError("Tensor parallel requires PyTorch >= 2.5.0")

    if sequence_parallel:
        # Special handling for Qwen3 QK Norm
        class Qwen3QKNorm(SequenceParallel):
            """Qwen3 Q/K LayerNorm, input is already sharded by head."""

            @staticmethod
            def _prepare_input_fn(sequence_sharding, mod, inputs, device_mesh):
                input_tensor = inputs[0]
                if isinstance(input_tensor, DTensor):
                    # Input is already DTensor, keep as is
                    assert input_tensor.placements == (Shard(dim=2),)
                    return input_tensor
                elif isinstance(input_tensor, torch.Tensor):
                    # Create DTensor from local tensor
                    return DTensor.from_local(
                        input_tensor,
                        device_mesh,
                        sequence_sharding,
                        run_check=False,
                    )
                else:
                    raise ValueError(f"Unexpected input type: {type(input_tensor)}")

        base_plan: Dict[str, ParallelStyle] = {
            "lm_head": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Shard(-1),
                use_local_output=False,
            ),
            "model.embed_tokens": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "model.norm": SequenceParallel(),
            "model.layers.*.input_layernorm": SequenceParallelAllGatherActivation(),
            "model.layers.*.self_attn.q_proj": ColwiseParallel(),
            "model.layers.*.self_attn.k_proj": ColwiseParallel(),
            "model.layers.*.self_attn.v_proj": ColwiseParallel(),
            "model.layers.*.self_attn.qkv_proj": ColwiseParallel(),
            "model.layers.*.self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
            "model.layers.*.self_attn.q_norm": Qwen3QKNorm(),  # Qwen3 specific
            "model.layers.*.self_attn.k_norm": Qwen3QKNorm(),  # Qwen3 specific
            "model.layers.*.post_attention_layernorm": SequenceParallelAllGatherActivation(),
            "model.layers.*.mlp.up_proj": ColwiseParallel(),
            "model.layers.*.mlp.gate_proj": ColwiseParallel(),
            "model.layers.*.mlp.gate_up_proj": ColwiseParallel(),
            "model.layers.*.mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
        }
    else:
        base_plan = {
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

    return base_plan


# Model class to TP plan function mapping
# Using string class names to avoid import dependencies
PARALLELIZE_FUNCTIONS: Dict[str, Callable[..., Dict[str, "ParallelStyle"]]] = {
    # LLaMA series
    "LlamaForCausalLM": _parallelize_llama,
    "LlamaForSequenceClassification": _parallelize_llama,
    # Qwen series
    "Qwen2ForCausalLM": _parallelize_qwen,
    "Qwen3ForCausalLM": _parallelize_qwen,
    "Qwen2ForSequenceClassification": _parallelize_qwen,
    "Qwen3ForSequenceClassification": _parallelize_qwen,
    # Mistral (similar to LLaMA)
    "MistralForCausalLM": _parallelize_llama,
}


def get_optimized_tp_plan(
    model: nn.Module,
    sequence_parallel: bool = False,
) -> Optional[Dict[str, "ParallelStyle"]]:
    """
    Get optimized TP plan for a model.

    If the model type is registered in PARALLELIZE_FUNCTIONS, returns the optimized plan;
    otherwise returns None, and the caller should use base plan or HF plan.

    Args:
        model: Model to parallelize
        sequence_parallel: Whether to enable Sequence Parallel

    Returns:
        Optimized TP plan dictionary, or None if not supported
    """
    model_cls_name = type(model).__name__

    if model_cls_name in PARALLELIZE_FUNCTIONS:
        try:
            plan_fn = PARALLELIZE_FUNCTIONS[model_cls_name]
            plan = plan_fn(model, sequence_parallel)
            logger.info(f"[optimized_tp_plans] Using optimized TP plan for {model_cls_name}")
            return plan
        except Exception as e:
            logger.warning(f"[optimized_tp_plans] Failed to generate optimized plan for {model_cls_name}: {e}")
            return None

    return None


def register_tp_plan(
    model_cls_name: str,
    plan_fn: Callable[..., Dict[str, "ParallelStyle"]],
) -> None:
    """
    Register a custom model's TP plan function.

    Args:
        model_cls_name: Model class name (string)
        plan_fn: Function to generate TP plan, signature: (model, sequence_parallel) -> Dict[str, ParallelStyle]
    """
    if model_cls_name in PARALLELIZE_FUNCTIONS:
        logger.warning(f"[optimized_tp_plans] Overwriting existing plan for {model_cls_name}")
    PARALLELIZE_FUNCTIONS[model_cls_name] = plan_fn
    logger.info(f"[optimized_tp_plans] Registered TP plan for {model_cls_name}")
