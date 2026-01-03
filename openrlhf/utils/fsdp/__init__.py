"""
FSDP2 Module for OpenRLHF
=========================

Requires: PyTorch >= 2.7.0

Files:
- strategy.py   : FSDP2Strategy class (main entry point)
- tp.py         : Tensor Parallelism
- checkpoint.py : Save/load checkpoints
- utils.py      : Gradient clipping, EMA, optimizer utils
"""

import torch
from packaging import version

# Fail fast if PyTorch version is too old
_torch_ver = version.parse(torch.__version__.split("+")[0])
if _torch_ver < version.parse("2.7.0"):
    raise RuntimeError(f"FSDP2 requires PyTorch >= 2.7.0, found {torch.__version__}")

from .checkpoint import load_distributed_checkpoint, load_hf_model, save_distributed_checkpoint, save_hf_model

# Public API
from .strategy import FSDP2Strategy
from .tp import apply_tensor_parallel, get_tp_plan, validate_tp_mesh
from .utils import (
    get_runtime_metadata,
    move_optimizer_state,
    moving_average_fsdp,
)

__all__ = [
    # Core
    "FSDP2Strategy",
    # Tensor Parallelism
    "apply_tensor_parallel",
    "get_tp_plan",
    "validate_tp_mesh",
    # Checkpointing
    "save_hf_model",
    "load_hf_model",
    "save_distributed_checkpoint",
    "load_distributed_checkpoint",
    # Utilities
    "moving_average_fsdp",
    "move_optimizer_state",
    "get_runtime_metadata",
]
