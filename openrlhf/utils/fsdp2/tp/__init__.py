"""
Tensor Parallelism Module for FSDP2
===================================

Files:
- tp_parallel.py : TP styles, plans, apply_tensor_parallel, Ring Attention compat
- tp_loss.py  : TP-aware loss computation (log_probs, entropy, cross_entropy)
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

from .tp_loss import (
    # Unified API (auto TP detection) - recommended
    compute_entropy,
    cross_entropy_loss,
    cross_entropy_loss_with_acc,
    log_probs_from_logits,
    prepare_vocab_parallel_logits,
    select_token_logits,
    # Low-level vocab parallel (advanced)
    vocab_parallel_cross_entropy,
    vocab_parallel_entropy,
    vocab_parallel_logprobs,
    vocab_parallel_logprobs_entropy,
)

__all__ = [
    # Parallel Styles
    "ColwiseParallelLora",
    "RowwiseParallelLora",
    "ReplicateParallel",
    "SequenceParallelPreserveGrad",
    # TP Plan Functions
    "apply_tensor_parallel",
    "validate_tp_mesh",
    # Ring Attention Compat
    "AttentionDTensorHook",
    "register_attention_hooks",
    # Unified API (auto TP detection)
    "log_probs_from_logits",
    "compute_entropy",
    "cross_entropy_loss",
    "cross_entropy_loss_with_acc",
    "prepare_vocab_parallel_logits",
    "select_token_logits",
    # Low-level Vocab Parallel (advanced)
    "vocab_parallel_logprobs",
    "vocab_parallel_logprobs_entropy",
    "vocab_parallel_entropy",
    "vocab_parallel_cross_entropy",
    # Monkey Patch
    "patch_for_tp",
]
