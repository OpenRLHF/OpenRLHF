"""
FSDP2 module for OpenRLHF. Requires PyTorch >= 2.7.0

Structure:
- strategy.py   : Core FSDP2Strategy class
- tp.py         : Tensor Parallelism (plans, styles, application)
- checkpoint.py : Save/load checkpoints
- utils.py      : Utilities (gradient clipping, EMA, optimizer state)
"""

import torch
from packaging import version

# Version check at import time
_v = version.parse(torch.__version__.split("+")[0])
if _v < version.parse("2.7.0"):
    raise RuntimeError(f"FSDP2 requires PyTorch >= 2.7.0, found {torch.__version__}")

# Mesh dimension names
MESH_DIM_DP = "dp"
MESH_DIM_CP = "cp"
MESH_DIM_TP = "tp"

from .checkpoint import (
    load_distributed_checkpoint,
    load_hf_model,
    save_distributed_checkpoint,
    save_hf_model,
)
from .strategy import FSDP2Strategy
from .tp import apply_tensor_parallel, get_tp_plan, validate_tp_mesh
from .utils import (
    barrier,
    clip_grad_norm_dtensor,
    get_runtime_metadata,
    move_optimizer_state,
    moving_average_fsdp,
    unwrap_actor,
)

__all__ = [
    "FSDP2Strategy",
    "apply_tensor_parallel",
    "get_tp_plan",
    "validate_tp_mesh",
    "save_hf_model",
    "load_hf_model",
    "save_distributed_checkpoint",
    "load_distributed_checkpoint",
    "barrier",
    "clip_grad_norm_dtensor",
    "move_optimizer_state",
    "moving_average_fsdp",
    "unwrap_actor",
    "get_runtime_metadata",
    "MESH_DIM_DP",
    "MESH_DIM_CP",
    "MESH_DIM_TP",
]
