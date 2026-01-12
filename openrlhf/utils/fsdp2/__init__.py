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
        tp_loss.py  - TP-aware loss (vocab-parallel log_probs, entropy, cross_entropy)
"""

# Core
from .strategy import FSDP2Strategy

# Tensor Parallelism
from .tp import (
    apply_tensor_parallel,
    cross_entropy_loss,
    cross_entropy_loss_with_acc,
    get_tp_plan,
    log_probs_from_logits,
    validate_tp_mesh,
    vocab_parallel_cross_entropy,
    vocab_parallel_entropy,
    vocab_parallel_logprobs,
    vocab_parallel_logprobs_entropy,
)

# Checkpointing
from .checkpoint import (
    load_distributed_checkpoint,
    load_hf_model,
    save_distributed_checkpoint,
    save_hf_model,
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
    "cross_entropy_loss",
    "cross_entropy_loss_with_acc",
    "get_tp_plan",
    "log_probs_from_logits",
    "validate_tp_mesh",
    "vocab_parallel_cross_entropy",
    "vocab_parallel_entropy",
    "vocab_parallel_logprobs",
    "vocab_parallel_logprobs_entropy",
    # Checkpointing
    "load_distributed_checkpoint",
    "load_hf_model",
    "save_distributed_checkpoint",
    "save_hf_model",
    # Utilities
    "get_checkpoint_metadata",
    "move_optimizer_state",
    "moving_average_fsdp2",
]
