"""Tensor Parallelism Module for FSDP2.

Files:
- tp_parallel.py : TP styles, plans, apply_tensor_parallel
- loss_parallel.py : Loss computation on vocab-sharded DTensor logits
"""

from .loss_parallel import (
    compute_entropy_sharded,
    compute_kd_loss_sharded,
    compute_token_log_probs_sharded,
    gather_token_logits_sharded,
    compute_argmax_sharded,
)
from .tp_parallel import (
    ReplicateParallel,
    SequenceParallelPreserveGrad,
    apply_tensor_parallel,
    get_tp_plan,
    validate_tp_mesh,
)

__all__ = [
    # Parallel Styles
    "ReplicateParallel",
    "SequenceParallelPreserveGrad",
    # TP Plan Functions
    "apply_tensor_parallel",
    "get_tp_plan",
    "validate_tp_mesh",
    # Loss Parallel
    "compute_token_log_probs_sharded",
    "compute_entropy_sharded",
    "gather_token_logits_sharded",
    "compute_kd_loss_sharded",
    "compute_argmax_sharded",
]
