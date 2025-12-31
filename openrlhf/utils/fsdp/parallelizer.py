# Copyright (c) 2025, OpenRLHF Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tensor Parallelism utilities using PyTorch native parallelize_module API.

Based on NeMo Automodel implementation, uses PyTorch native TP API instead of
transformers built-in implementation:
- More reliable training support
- Non-TP parameters are unaffected (no gradient sync issues)
- use_local_output=True ensures activations are passed as regular Tensors

Usage:
    from openrlhf.utils.fsdp.parallelizer import apply_tensor_parallel, get_tp_plan

    # 1. Load model (without TP)
    model = AutoModelForCausalLM.from_pretrained(...)

    # 2. Get TP plan
    tp_plan = get_tp_plan(model)

    # 3. Apply TP
    apply_tensor_parallel(model, tp_mesh, tp_plan)

    # 4. Apply FSDP
    model = fully_shard(model, ...)
"""

import logging
import re
from functools import lru_cache
from typing import Dict, Optional, Union

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
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor.parallel import (
        ColwiseParallel,
        ParallelStyle,
        RowwiseParallel,
        SequenceParallel,
        parallelize_module,
    )
    from torch.distributed.tensor.placement_types import Replicate, Shard


@lru_cache
def translate_to_torch_parallel_style(style: str) -> "ParallelStyle":
    """
    Convert string description to PyTorch ParallelStyle object.

    Args:
        style: Parallel style string, e.g., "colwise", "rowwise", etc.

    Returns:
        Corresponding ParallelStyle object
    """
    if not _is_torch_tp_available():
        raise RuntimeError("Tensor parallel requires PyTorch >= 2.5.0")

    if style == "colwise":
        return ColwiseParallel()
    elif style == "rowwise":
        return RowwiseParallel()
    elif style == "colwise_rep":
        return ColwiseParallel(output_layouts=Replicate())
    elif style == "rowwise_rep":
        return RowwiseParallel(input_layouts=Replicate())
    elif style == "sequence_parallel":
        return SequenceParallel()
    else:
        raise ValueError(f"Unknown parallel style: {style}")


def get_base_tp_plan(sequence_parallel: bool = False) -> Dict[str, "ParallelStyle"]:
    """
    Get generic base TP plan, suitable for LLaMA-style models.

    Args:
        sequence_parallel: Whether to enable sequence parallel

    Returns:
        TP plan dictionary
    """
    if not _is_torch_tp_available():
        raise RuntimeError("Tensor parallel requires PyTorch >= 2.5.0")

    base_plan = {
        "model.embed_tokens": RowwiseParallel(input_layouts=Replicate()),
        "model.layers.*.self_attn.q_proj": ColwiseParallel(),
        "model.layers.*.self_attn.k_proj": ColwiseParallel(),
        "model.layers.*.self_attn.v_proj": ColwiseParallel(),
        "model.layers.*.self_attn.qkv_proj": ColwiseParallel(),  # Combined QKV
        "model.layers.*.self_attn.o_proj": RowwiseParallel(),
        "model.layers.*.mlp.gate_up_proj": ColwiseParallel(),  # Fused gate+up
        "model.layers.*.mlp.up_proj": ColwiseParallel(),
        "model.layers.*.mlp.gate_proj": ColwiseParallel(),
        "model.layers.*.mlp.down_proj": RowwiseParallel(),
        # lm_head uses Shard(-1) to reduce communication overhead, use_local_output=False keeps DTensor
        "lm_head": ColwiseParallel(output_layouts=Shard(-1), use_local_output=False),
    }

    if sequence_parallel:
        sp_plan = {
            "model.embed_tokens": RowwiseParallel(input_layouts=Replicate(), output_layouts=Shard(1)),
            "model.norm": SequenceParallel(),
            "model.layers.*.input_layernorm": SequenceParallel(),
            "model.layers.*.self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
            "model.layers.*.post_attention_layernorm": SequenceParallel(),
            "model.layers.*.mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
            "lm_head": ColwiseParallel(input_layouts=Shard(1), output_layouts=Replicate()),
        }
        base_plan.update(sp_plan)

    return base_plan


def get_hf_tp_plan(model: nn.Module) -> Dict[str, "ParallelStyle"]:
    """
    Get TP plan from HuggingFace model.

    Priority:
    1. model_cls._tp_plan
    2. model._tp_plan
    3. model.model._tp_plan (inner model)

    Args:
        model: HuggingFace model

    Returns:
        TP plan dictionary
    """
    if not _is_torch_tp_available():
        raise RuntimeError("Tensor parallel requires PyTorch >= 2.5.0")

    model_cls = type(model)
    hf_tp_plan = {}

    # Get from model class
    if hasattr(model_cls, "_tp_plan") and model_cls._tp_plan is not None:
        hf_tp_plan.update(model_cls._tp_plan)

    # Get from model instance
    if hasattr(model, "_tp_plan") and model._tp_plan is not None:
        hf_tp_plan.update(model._tp_plan)

    # Get from inner model
    if hasattr(model, "model") and hasattr(model.model, "_tp_plan") and model.model._tp_plan is not None:
        for k, v in model.model._tp_plan.items():
            hf_tp_plan[f"model.{k}"] = v

    if not hf_tp_plan:
        return {}

    # Add embed_tokens (if missing)
    if "model.embed_tokens" not in hf_tp_plan:
        hf_tp_plan["model.embed_tokens"] = "rowwise_rep"

    # Convert strings to ParallelStyle objects
    result = {}
    for k, v in hf_tp_plan.items():
        if isinstance(v, str):
            try:
                result[k] = translate_to_torch_parallel_style(v)
            except ValueError as e:
                logger.warning(f"Skipping unknown parallel style for {k}: {e}")
        else:
            result[k] = v

    # lm_head uses Shard(-1) to reduce communication overhead, use_local_output=False keeps DTensor
    if "lm_head" in result and isinstance(result["lm_head"], ColwiseParallel):
        result["lm_head"] = ColwiseParallel(output_layouts=Shard(-1), use_local_output=False)

    logger.info(f"[parallelizer] HF TP plan: {list(result.keys())}")
    return result


def get_tp_plan(
    model: nn.Module,
    sequence_parallel: bool = False,
    use_hf_tp_plan: bool = True,
    custom_plan: Optional[Dict[str, Union[str, "ParallelStyle"]]] = None,
) -> Dict[str, "ParallelStyle"]:
    """
    Get TP plan for a model.

    Priority:
    1. custom_plan (if provided)
    2. Optimized model-specific plan (if available)
    3. HF model's _tp_plan (if use_hf_tp_plan=True)
    4. Generic base plan

    Args:
        model: Model
        sequence_parallel: Whether to enable sequence parallel
        use_hf_tp_plan: Whether to try getting TP plan from HF model
        custom_plan: Custom TP plan

    Returns:
        TP plan dictionary
    """
    if not _is_torch_tp_available():
        raise RuntimeError("Tensor parallel requires PyTorch >= 2.5.0")

    # 1. Use custom plan
    if custom_plan is not None:
        result = {}
        for k, v in custom_plan.items():
            if isinstance(v, str):
                result[k] = translate_to_torch_parallel_style(v)
            else:
                result[k] = v
        logger.info("[parallelizer] Using custom TP plan")
        return result

    # 2. Try using optimized model-specific plan
    try:
        from openrlhf.utils.fsdp.optimized_tp_plans import get_optimized_tp_plan

        optimized_plan = get_optimized_tp_plan(model, sequence_parallel)
        if optimized_plan:
            logger.info("[parallelizer] Using optimized TP plan")
            return optimized_plan
    except ImportError:
        pass

    # 3. Try getting from HF
    if use_hf_tp_plan:
        hf_plan = get_hf_tp_plan(model)
        if hf_plan:
            logger.info("[parallelizer] Using HF TP plan")
            return hf_plan

    # 4. Use base plan
    logger.info("[parallelizer] Using base TP plan (LLaMA-style)")
    return get_base_tp_plan(sequence_parallel)


def validate_tp_mesh(model: nn.Module, tp_mesh: "DeviceMesh") -> None:
    """
    Validate TP mesh compatibility with model.

    Check if attention heads are divisible by TP size.

    Args:
        model: Model
        tp_mesh: TP device mesh

    Raises:
        AssertionError: If incompatible
    """
    if tp_mesh.size() == 1:
        return

    config = getattr(model, "config", None)
    if config is None:
        logger.warning("[parallelizer] Model has no config, skipping TP validation")
        return

    num_attention_heads = getattr(config, "num_attention_heads", 0)
    num_key_value_heads = getattr(config, "num_key_value_heads", num_attention_heads)

    if num_attention_heads > 0:
        assert num_attention_heads % tp_mesh.size() == 0, (
            f"num_attention_heads ({num_attention_heads}) must be divisible by " f"TP size ({tp_mesh.size()})"
        )

    if num_key_value_heads > 0:
        assert num_key_value_heads % tp_mesh.size() == 0, (
            f"num_key_value_heads ({num_key_value_heads}) must be divisible by " f"TP size ({tp_mesh.size()})"
        )

    logger.info(
        f"[parallelizer] TP validation passed: heads={num_attention_heads}, "
        f"kv_heads={num_key_value_heads}, tp_size={tp_mesh.size()}"
    )


def _expand_wildcard_plan(plan: Dict[str, "ParallelStyle"], model: nn.Module) -> Dict[str, "ParallelStyle"]:
    """
    Expand wildcards (*) in plan to specific module names.

    Args:
        plan: TP plan containing wildcards
        model: Model

    Returns:
        Expanded TP plan
    """
    expanded = {}
    module_names = {name for name, _ in model.named_modules()}

    for pattern, style in plan.items():
        if "*" not in pattern:
            # No wildcard, use directly
            if pattern in module_names or pattern == "":
                expanded[pattern] = style
            continue

        # Convert pattern to regex
        # "model.layers.*.self_attn.q_proj" -> "model\.layers\.\d+\.self_attn\.q_proj"
        regex_pattern = pattern.replace(".", r"\.").replace("*", r"\d+")
        regex = re.compile(f"^{regex_pattern}$")

        for name in module_names:
            if regex.match(name):
                expanded[name] = style

    logger.debug(f"[parallelizer] Expanded plan: {len(plan)} patterns -> {len(expanded)} modules")
    return expanded


def apply_tensor_parallel(
    model: nn.Module,
    tp_mesh: "DeviceMesh",
    tp_plan: Optional[Dict[str, "ParallelStyle"]] = None,
    sequence_parallel: bool = False,
    validate: bool = True,
) -> nn.Module:
    """
    Apply Tensor Parallelism to model.

    Uses PyTorch native parallelize_module API.

    Args:
        model: Model
        tp_mesh: TP device mesh
        tp_plan: TP plan, auto-generated if None
        sequence_parallel: Whether to enable sequence parallel
        validate: Whether to validate TP compatibility

    Returns:
        Model with TP applied
    """
    if not _is_torch_tp_available():
        raise RuntimeError("Tensor parallel requires PyTorch >= 2.5.0")

    if tp_mesh.size() == 1:
        logger.info("[parallelizer] TP size is 1, skipping tensor parallel")
        return model

    # Validate compatibility
    if validate:
        validate_tp_mesh(model, tp_mesh)

    # Get TP plan
    if tp_plan is None:
        tp_plan = get_tp_plan(model, sequence_parallel=sequence_parallel)

    # Always apply LoRA translation (no-op for non-LoRA layers)
    # This follows Automodel's approach where translate_to_lora is called unconditionally
    try:
        from openrlhf.utils.fsdp.parallel_styles import translate_to_lora

        tp_plan = {k: translate_to_lora(v) for k, v in tp_plan.items()}
    except ImportError:
        pass

    if not tp_plan:
        logger.warning("[parallelizer] Empty TP plan, skipping tensor parallel")
        return model

    # Expand wildcards
    expanded_plan = _expand_wildcard_plan(tp_plan, model)

    if not expanded_plan:
        logger.warning("[parallelizer] No modules matched TP plan patterns")
        return model

    logger.info(f"[parallelizer] Applying TP to {len(expanded_plan)} modules")

    # Apply TP
    parallelize_module(model, tp_mesh, expanded_plan)

    return model
