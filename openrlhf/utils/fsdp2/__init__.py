"""
FSDP2 Module for OpenRLHF
=========================

Provides FSDP2 training strategy with Tensor Parallelism (TP) support.

Structure:
    strategy.py   - FSDP2Strategy (main entry point)
    checkpoint.py - Distributed checkpoint save/load
    utils.py      - EMA, optimizer state management
    tp/           - Tensor Parallelism
        tp_parallel.py - TP styles, plans, apply_tensor_parallel

Note:
    Import FSDP2Strategy from `openrlhf.utils.fsdp2.strategy` to avoid
    circular imports during package initialization.
"""

# Tensor Parallelism
from .tp import (
    apply_tensor_parallel,
    get_tp_plan,
    validate_tp_mesh,
)

# Utilities
from .utils import (
    get_checkpoint_metadata,
    move_optimizer_state,
    moving_average_fsdp2,
)

__all__ = [
    # Tensor Parallelism
    "apply_tensor_parallel",
    "get_tp_plan",
    "validate_tp_mesh",
    # Utilities
    "get_checkpoint_metadata",
    "move_optimizer_state",
    "moving_average_fsdp2",
]
