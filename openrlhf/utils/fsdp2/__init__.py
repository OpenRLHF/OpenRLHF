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
"""

# Checkpointing
from .checkpoint import (
    load_hf_weights,
    save_hf_checkpoint,
    load_dcp_checkpoint,
    save_dcp_checkpoint,
)

# Core
from .strategy import FSDP2Strategy

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
    # Core
    "FSDP2Strategy",
    # Tensor Parallelism
    "apply_tensor_parallel",
    "get_tp_plan",
    "validate_tp_mesh",
    # Checkpointing
    "load_hf_weights",
    "save_hf_checkpoint",
    "load_dcp_checkpoint",
    "save_dcp_checkpoint",
    # Utilities
    "get_checkpoint_metadata",
    "move_optimizer_state",
    "moving_average_fsdp2",
]
