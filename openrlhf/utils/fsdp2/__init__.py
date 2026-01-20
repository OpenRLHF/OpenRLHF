"""FSDP2 Strategy module for OpenRLHF.

This module provides FSDP2-based training as an alternative to DeepSpeed.
It includes support for:
- Pure FSDP2 (data parallelism with fully sharded parameters)
- FSDP2 + AutoTP (tensor parallelism using HuggingFace's built-in ._tp_plan)
"""

from .fsdp2 import FSDP2Strategy
from .fsdp2_utils import (
    clip_grad_by_total_norm_,
    get_gemma_tp_plan,
    get_grad_norm,
    get_hf_tp_plan,
    get_llama_tp_plan,
    get_optimized_tp_plan,
    get_optimizer_grouped_parameters,
    get_qwen_tp_plan,
    to_local_if_dtensor,
    translate_parallel_style,
)

__all__ = [
    # Main strategy class
    "FSDP2Strategy",
    # Optimizer utilities
    "get_optimizer_grouped_parameters",
    # Gradient utilities
    "get_grad_norm",
    "clip_grad_by_total_norm_",
    # DTensor utilities
    "to_local_if_dtensor",
    # Tensor parallel plans
    "get_llama_tp_plan",
    "get_hf_tp_plan",
    "get_optimized_tp_plan",
    "get_qwen_tp_plan",
    "get_gemma_tp_plan",
    "translate_parallel_style",
]
