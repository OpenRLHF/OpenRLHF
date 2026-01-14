"""
Tensor Parallelism Module for FSDP2
===================================

Files:
- tp_parallel.py : TP styles, plans, apply_tensor_parallel, Ring Attention compat
"""

from .tp_parallel import (
    # Parallel Styles
    ColwiseParallelLora,
    ReplicateParallel,
    RowwiseParallelLora,
    SequenceParallelPreserveGrad,
    # TP Plan Functions
    apply_tensor_parallel,
    get_tp_plan,
    validate_tp_mesh,
    # Ring Attention Compat
    AttentionDTensorHook,
    register_attention_hooks,
)

__all__ = [
    # Parallel Styles
    "ColwiseParallelLora",
    "RowwiseParallelLora",
    "ReplicateParallel",
    "SequenceParallelPreserveGrad",
    # TP Plan Functions
    "apply_tensor_parallel",
    "get_tp_plan",
    "validate_tp_mesh",
    # Ring Attention Compat
    "AttentionDTensorHook",
    "register_attention_hooks",
]
