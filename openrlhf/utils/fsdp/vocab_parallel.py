"""
Vocab Parallel Loss Computation for Tensor Parallelism
=======================================================

This module provides efficient loss computation when logits are sharded on
the vocabulary dimension (Shard(-1)) in Tensor Parallelism.

Key Features:
- Avoids all-gathering the full vocab dimension across TP ranks
- Computes log probabilities and entropy directly on sharded logits
- Memory-efficient chunked processing for long sequences
- Compatible with DTensor's Shard(-1) placement

Reference: AReaL (areal/utils/functional/vocab_parallel.py)
"""

import functools
from typing import Callable, TypeVar

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

T = TypeVar("T", torch.Tensor, tuple[torch.Tensor, torch.Tensor])


# =============================================================================
# Non-Parallel Implementations (for TP size = 1)
# =============================================================================


def _gather_logprobs(logits: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Compute log probabilities for single-GPU case."""
    log_probs = F.log_softmax(logits.float() / temperature, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return log_probs_labels


def _gather_logprobs_entropy(
    logits: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute log probabilities and entropy for single-GPU case."""
    log_probs = F.log_softmax(logits.float() / temperature, dim=-1)
    entropy = -torch.sum(log_probs.exp() * log_probs, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return log_probs_labels, entropy


# torch.compile for non-parallel versions
try:
    _gather_logprobs_compiled = torch.compile(_gather_logprobs)
    _gather_logprobs_entropy_compiled = torch.compile(_gather_logprobs_entropy)
except Exception:
    _gather_logprobs_compiled = _gather_logprobs
    _gather_logprobs_entropy_compiled = _gather_logprobs_entropy


# =============================================================================
# Chunked Processing for Memory Efficiency
# =============================================================================


def _chunked_apply(
    fn: Callable[[torch.Tensor, torch.Tensor], T],
    logits: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: int = 1024,
) -> T:
    """Apply a function in chunks along the first dimension to reduce peak memory."""
    total_seqlen = logits.shape[0]
    assert total_seqlen > 0, "Input logits must have at least one element"
    results: list = []

    for i in range(0, total_seqlen, chunk_size):
        end_idx = min(i + chunk_size, total_seqlen)
        chunk_result = fn(logits[i:end_idx], labels[i:end_idx])
        results.append(chunk_result)

    # Handle single tensor vs tuple of tensors
    if isinstance(results[0], tuple):
        num_outputs = len(results[0])
        return tuple(torch.cat([r[i] for r in results]) for i in range(num_outputs))
    return torch.cat(results)


# =============================================================================
# Vocab Parallel Log Probabilities
# =============================================================================


class _VocabParallelLogProbs(torch.autograd.Function):
    """Compute log probabilities when logits are sharded on the vocab dimension.

    Given sharded logits [..., vocab_size/tp] and labels [...], computes:
        logprobs[i] = logits[i, labels[i]] - log(sum(exp(logits[i, :])))

    The input can have arbitrary leading dimensions (e.g., [batch, seq_len] or just
    [seq_len]). The labels indices are global (0 to vocab_size-1), and each TP rank
    only holds a partition of the vocabulary.

    Memory Optimization:
        Following Megatron's cross_entropy pattern, we use in-place operations to
        minimize memory allocations. The key optimization is in backward():

        - The gradient formula is: grad = one_hot(labels) - softmax
        - Since this only requires subtracting 1 at the label position and scaling,
          we can directly reuse the saved softmax tensor as grad_input (in-place).
        - This avoids allocating a new [*, vocab/tp] tensor for gradients.

    Note:
        This implementation uses in-place operations on saved tensors for memory
        efficiency. As a result, it does NOT support:
        - `retain_graph=True` in backward()
        - Higher-order gradients (e.g., torch.autograd.grad with create_graph=True)

    Reference: AReaL (areal/utils/functional/vocab_parallel.py)
    """

    @staticmethod
    def forward(
        ctx,
        vocab_parallel_logits: torch.Tensor,
        labels: torch.Tensor,
        tp_group: dist.ProcessGroup,
    ) -> torch.Tensor:
        # Get TP rank info
        tp_rank = dist.get_rank(tp_group)

        # Calculate vocab partition boundaries for this rank
        partition_vocab_size = vocab_parallel_logits.size(-1)
        vocab_start_index = tp_rank * partition_vocab_size
        vocab_end_index = vocab_start_index + partition_vocab_size

        # Step 1: Numerical stability - subtract max
        logits_max = vocab_parallel_logits.max(dim=-1, keepdim=True).values
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=tp_group)

        # Subtract max for numerical stability
        normalized_logits = vocab_parallel_logits - logits_max

        # Step 2: Compute exp and sum across all ranks
        exp_logits = normalized_logits.exp()
        sum_exp_logits = exp_logits.sum(dim=-1, keepdim=True)
        dist.all_reduce(sum_exp_logits, op=dist.ReduceOp.SUM, group=tp_group)

        # Step 3: Get the logit value at labels position
        labels_mask = (labels < vocab_start_index) | (labels >= vocab_end_index)
        masked_labels = labels.clone() - vocab_start_index
        masked_labels[labels_mask] = 0

        logits_2d = normalized_logits.view(-1, partition_vocab_size)
        masked_labels_1d = masked_labels.view(-1)
        arange_1d = torch.arange(logits_2d.size(0), device=logits_2d.device)

        predicted_logits_1d = logits_2d[arange_1d, masked_labels_1d]
        predicted_logits = predicted_logits_1d.view_as(labels)
        predicted_logits[labels_mask] = 0.0
        dist.all_reduce(predicted_logits, op=dist.ReduceOp.SUM, group=tp_group)

        # Step 4: Compute log probability
        log_sum_exp = sum_exp_logits.squeeze(-1).log()
        logprobs = predicted_logits - log_sum_exp

        # Step 5: Compute softmax in-place for backward (reuse exp_logits memory)
        softmax = exp_logits.div_(sum_exp_logits)
        ctx.save_for_backward(softmax, labels_mask, masked_labels_1d)

        return logprobs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        softmax, labels_mask, masked_labels_1d = ctx.saved_tensors

        # Gradient of logprobs w.r.t. logits: one_hot(labels) - softmax
        partition_vocab_size = softmax.size(-1)

        # Use softmax as the gradient base (will be modified)
        grad_input = softmax
        grad_2d = grad_input.view(-1, partition_vocab_size)
        arange_1d = torch.arange(grad_2d.size(0), device=grad_2d.device)

        # Subtract 1 at labels position (only for labels in this partition)
        update_mask = ~labels_mask.view(-1)
        grad_2d[arange_1d, masked_labels_1d] -= update_mask.float()

        # Scale by grad_output (in-place)
        # Note: we want -(softmax - one_hot) = one_hot - softmax for logprobs gradient
        grad_input.mul_(grad_output.unsqueeze(-1))
        grad_input.neg_()

        return grad_input, None, None


class _VocabParallelLogProbsEntropy(torch.autograd.Function):
    """Compute both log probabilities and entropy when logits are sharded.

    This combines the computation to share intermediate results (softmax, sum_exp, etc.)
    and reduce redundant all-reduce operations compared to calling logprobs and entropy
    separately.

    Memory Optimization:
        Forward saves only ONE large tensor (softmax) plus a few small scalars.
        Backward allocates ONE new large tensor for grad_input.

    Reference: AReaL (areal/utils/functional/vocab_parallel.py)
    """

    @staticmethod
    def forward(
        ctx,
        vocab_parallel_logits: torch.Tensor,
        labels: torch.Tensor,
        tp_group: dist.ProcessGroup,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Get TP rank info
        tp_rank = dist.get_rank(tp_group)
        partition_vocab_size = vocab_parallel_logits.size(-1)
        vocab_start_index = tp_rank * partition_vocab_size
        vocab_end_index = vocab_start_index + partition_vocab_size

        # Step 1: Numerical stability - subtract max
        logits_max = vocab_parallel_logits.max(dim=-1, keepdim=True).values
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=tp_group)

        normalized_logits = vocab_parallel_logits - logits_max

        # Step 2: Compute exp and sum_exp
        exp_logits = normalized_logits.exp()
        sum_exp_logits = exp_logits.sum(dim=-1, keepdim=True)
        dist.all_reduce(sum_exp_logits, op=dist.ReduceOp.SUM, group=tp_group)

        # Step 3: Compute softmax in-place
        softmax = exp_logits.div_(sum_exp_logits)

        # Step 4: For logprobs - get labels logit
        labels_mask = (labels < vocab_start_index) | (labels >= vocab_end_index)
        masked_labels = labels.clone() - vocab_start_index
        masked_labels[labels_mask] = 0

        logits_2d = normalized_logits.view(-1, partition_vocab_size)
        masked_labels_1d = masked_labels.view(-1)
        arange_1d = torch.arange(logits_2d.size(0), device=logits_2d.device)

        predicted_logits_1d = logits_2d[arange_1d, masked_labels_1d]
        predicted_logits = predicted_logits_1d.view_as(labels)
        predicted_logits[labels_mask] = 0.0
        dist.all_reduce(predicted_logits, op=dist.ReduceOp.SUM, group=tp_group)

        # Step 5: For entropy - compute sum(softmax * logits)
        sum_softmax_times_logits = (softmax * vocab_parallel_logits).sum(dim=-1, keepdim=True)
        dist.all_reduce(sum_softmax_times_logits, op=dist.ReduceOp.SUM, group=tp_group)

        # Step 6: Compute final results
        log_sum_exp = sum_exp_logits.log()
        logprobs = predicted_logits - log_sum_exp.squeeze(-1)
        # entropy = log(Z) - E[x] = (max + log(sum_exp)) - sum_softmax_times_logits
        entropy = (logits_max + log_sum_exp - sum_softmax_times_logits).squeeze(-1)

        # Compute log(Z) for backward
        log_z = logits_max + log_sum_exp

        # Save for backward
        ctx.save_for_backward(
            softmax,
            sum_softmax_times_logits,
            log_z,
            labels_mask,
            masked_labels_1d,
        )
        ctx.partition_vocab_size = partition_vocab_size

        return logprobs, entropy

    @staticmethod
    def backward(ctx, grad_logprobs: torch.Tensor, grad_entropy: torch.Tensor) -> tuple:
        (
            softmax,
            sum_softmax_times_logits,
            log_z,
            labels_mask,
            masked_labels_1d,
        ) = ctx.saved_tensors
        partition_vocab_size = ctx.partition_vocab_size

        # Compute entropy gradient contribution
        mean_x_minus_log_z = sum_softmax_times_logits - log_z

        # grad_input = softmax * (mean_x - log_z) - xlogy(softmax, softmax)
        grad_input = softmax * mean_x_minus_log_z
        grad_input.sub_(torch.xlogy(softmax, softmax))
        grad_input.mul_(grad_entropy.unsqueeze(-1))

        # Add logprobs gradient contribution
        grad_input.sub_(softmax * grad_logprobs.unsqueeze(-1))

        # Add one_hot * grad_logprobs at labels positions
        grad_2d = grad_input.view(-1, partition_vocab_size)
        arange_1d = torch.arange(grad_2d.size(0), device=grad_2d.device)
        update_mask = ~labels_mask.view(-1)
        grad_2d[arange_1d, masked_labels_1d] += update_mask * grad_logprobs.view(-1)

        return grad_input, None, None


# =============================================================================
# Vocab Parallel Cross Entropy
# =============================================================================


class _VocabParallelCrossEntropy(torch.autograd.Function):
    """Compute cross entropy loss when logits are sharded on the vocab dimension.

    This is equivalent to -logprobs, but optimized for loss computation
    where we don't need the logprobs as output.

    Reference: Megatron-LM's parallel cross entropy
    """

    @staticmethod
    def forward(
        ctx,
        vocab_parallel_logits: torch.Tensor,
        labels: torch.Tensor,
        tp_group: dist.ProcessGroup,
        ignore_index: int = -100,
    ) -> torch.Tensor:
        tp_rank = dist.get_rank(tp_group)
        partition_vocab_size = vocab_parallel_logits.size(-1)
        vocab_start_index = tp_rank * partition_vocab_size
        vocab_end_index = vocab_start_index + partition_vocab_size

        # Create mask for ignored indices
        ignore_mask = labels == ignore_index

        # Numerical stability - subtract max
        logits_max = vocab_parallel_logits.max(dim=-1, keepdim=True).values
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=tp_group)

        normalized_logits = vocab_parallel_logits - logits_max

        # Compute exp and sum
        exp_logits = normalized_logits.exp()
        sum_exp_logits = exp_logits.sum(dim=-1, keepdim=True)
        dist.all_reduce(sum_exp_logits, op=dist.ReduceOp.SUM, group=tp_group)

        # Get labels logit
        labels_mask = (labels < vocab_start_index) | (labels >= vocab_end_index) | ignore_mask
        masked_labels = labels.clone() - vocab_start_index
        masked_labels[labels_mask] = 0

        logits_2d = normalized_logits.view(-1, partition_vocab_size)
        masked_labels_1d = masked_labels.view(-1)
        arange_1d = torch.arange(logits_2d.size(0), device=logits_2d.device)

        predicted_logits_1d = logits_2d[arange_1d, masked_labels_1d]
        predicted_logits = predicted_logits_1d.view_as(labels)
        predicted_logits[labels_mask] = 0.0
        dist.all_reduce(predicted_logits, op=dist.ReduceOp.SUM, group=tp_group)

        # Compute cross entropy: log(sum_exp) - predicted_logit
        log_sum_exp = sum_exp_logits.squeeze(-1).log()
        loss = log_sum_exp - predicted_logits
        loss[ignore_mask] = 0.0

        # Compute softmax for backward
        softmax = exp_logits.div_(sum_exp_logits)
        ctx.save_for_backward(softmax, labels_mask, masked_labels_1d, ignore_mask)
        ctx.ignore_index = ignore_index

        return loss

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        softmax, labels_mask, masked_labels_1d, ignore_mask = ctx.saved_tensors
        partition_vocab_size = softmax.size(-1)

        # Gradient: softmax - one_hot(labels)
        grad_input = softmax.clone()
        grad_2d = grad_input.view(-1, partition_vocab_size)
        arange_1d = torch.arange(grad_2d.size(0), device=grad_2d.device)

        update_mask = ~labels_mask.view(-1)
        grad_2d[arange_1d, masked_labels_1d] -= update_mask.float()

        # Apply grad_output and mask ignored positions
        grad_input.mul_(grad_output.unsqueeze(-1))
        if ignore_mask.any():
            grad_input[ignore_mask.view(*ignore_mask.shape, 1).expand_as(grad_input)] = 0.0

        return grad_input, None, None, None


# =============================================================================
# Public API Functions
# =============================================================================


def _get_local_logits(logits: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Convert DTensor to local tensor if needed, return (tensor, was_dtensor)."""
    if isinstance(logits, DTensor):
        return logits.to_local(), True
    return logits, False


def vocab_parallel_logprobs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    tp_group: dist.ProcessGroup | None = None,
    temperature: float = 1.0,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """Compute log probabilities with optional vocab parallelism.

    Args:
        logits: Model logits with shape [..., vocab_size] or [..., vocab_size/tp]
            when tensor parallelism is enabled. Can be DTensor with Shard(-1).
        labels: Token indices with shape [...] for which to compute log probabilities.
        tp_group: If provided with tp_size > 1, uses vocab-parallel computation
            to avoid gathering the full vocab dimension across TP ranks.
        temperature: Softmax temperature scaling. Default is 1.0.
        chunk_size: Chunk size for memory-efficient processing along the sequence
            dimension. Default is 1024.

    Returns:
        Log probabilities at the label positions with shape [...].
    """
    # Handle DTensor input
    local_logits, was_dtensor = _get_local_logits(logits)

    # Apply temperature
    if temperature != 1.0:
        local_logits = local_logits.float() / temperature
    else:
        local_logits = local_logits.float()

    # Use vocab-parallel if TP enabled
    if tp_group is not None and dist.get_world_size(tp_group) > 1:
        fn = functools.partial(_VocabParallelLogProbs.apply, tp_group=tp_group)
        return _chunked_apply(fn, local_logits, labels, chunk_size)

    # Single GPU path
    fn = functools.partial(_gather_logprobs_compiled, temperature=1.0)  # temperature already applied
    return _chunked_apply(fn, local_logits, labels, chunk_size)


def vocab_parallel_logprobs_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    tp_group: dist.ProcessGroup | None = None,
    temperature: float = 1.0,
    chunk_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute log probabilities and entropy with optional vocab parallelism.

    This function computes both values in a single pass, sharing intermediate results
    to reduce redundant computation and all-reduce operations.

    Args:
        logits: Model logits with shape [..., vocab_size] or [..., vocab_size/tp].
            Can be DTensor with Shard(-1).
        labels: Token indices with shape [...] for which to compute log probabilities.
        tp_group: If provided with tp_size > 1, uses vocab-parallel computation.
        temperature: Softmax temperature scaling. Default is 1.0.
        chunk_size: Chunk size for memory-efficient processing. Default is 1024.

    Returns:
        A tuple of (logprobs, entropy):
            - logprobs: Log probabilities at the label positions with shape [...].
            - entropy: Entropy of the probability distribution with shape [...].
    """
    # Handle DTensor input
    local_logits, was_dtensor = _get_local_logits(logits)

    # Apply temperature
    if temperature != 1.0:
        local_logits = local_logits.float() / temperature
    else:
        local_logits = local_logits.float()

    # Use vocab-parallel if TP enabled
    if tp_group is not None and dist.get_world_size(tp_group) > 1:
        fn = functools.partial(_VocabParallelLogProbsEntropy.apply, tp_group=tp_group)
        return _chunked_apply(fn, local_logits, labels, chunk_size)

    # Single GPU path
    fn = functools.partial(_gather_logprobs_entropy_compiled, temperature=1.0)
    return _chunked_apply(fn, local_logits, labels, chunk_size)


def vocab_parallel_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    tp_group: dist.ProcessGroup | None = None,
    ignore_index: int = -100,
    reduction: str = "none",
    chunk_size: int = 1024,
) -> torch.Tensor:
    """Compute cross entropy loss with optional vocab parallelism.

    Args:
        logits: Model logits with shape [..., vocab_size] or [..., vocab_size/tp].
            Can be DTensor with Shard(-1).
        labels: Target labels with shape [...].
        tp_group: If provided with tp_size > 1, uses vocab-parallel computation.
        ignore_index: Target value that is ignored. Default is -100.
        reduction: 'none' | 'mean' | 'sum'. Default is 'none'.
        chunk_size: Chunk size for memory-efficient processing. Default is 1024.

    Returns:
        Cross entropy loss. Shape depends on reduction.
    """
    # Handle DTensor input
    local_logits, was_dtensor = _get_local_logits(logits)
    local_logits = local_logits.float()

    # Use vocab-parallel if TP enabled
    if tp_group is not None and dist.get_world_size(tp_group) > 1:
        fn = functools.partial(
            _VocabParallelCrossEntropy.apply,
            tp_group=tp_group,
            ignore_index=ignore_index,
        )
        loss = _chunked_apply(fn, local_logits, labels, chunk_size)
    else:
        # Single GPU path - use standard cross entropy
        loss = F.cross_entropy(
            local_logits.view(-1, local_logits.size(-1)),
            labels.view(-1),
            ignore_index=ignore_index,
            reduction="none",
        ).view_as(labels)

    # Apply reduction
    if reduction == "none":
        return loss
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "mean":
        valid_tokens = (labels != ignore_index).sum()
        return loss.sum() / valid_tokens.clamp(min=1)
    else:
        raise ValueError(f"Invalid reduction: {reduction}")
