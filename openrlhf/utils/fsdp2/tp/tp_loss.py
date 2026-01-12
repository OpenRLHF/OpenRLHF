"""
TP-Aware Loss Computation
=========================

This module provides Tensor Parallel (TP) aware loss functions, specifically for vocab-parallel scenarios.
It includes:
1. Low-level autograd functions for vocab-parallel logits.
2. Public APIs for vocab-parallel loss computation.
3. Unified APIs that handle both TP (sharded) and non-TP (full) logits automatically.

Memory Optimization:
- Supports chunked computation (`chunk_size`) to prevent OOM on long sequences.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.tensor import DTensor, Shard
from typing import Callable, Tuple, Optional, List, Union


def _chunked_apply(
    fn: Callable[..., torch.Tensor | tuple[torch.Tensor, torch.Tensor]],
    *args: torch.Tensor,
    chunk_size: int = 1024,
    **kwargs,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Apply a function in chunks along the sequence dimension (dim 0) to reduce peak memory usage.
    Useful for preventing OOM when processing very long sequences.
    """
    if not args:
        return fn(*args, **kwargs)

    # Assuming the first argument determines the sequence length to chunk
    total_seqlen = args[0].shape[0]
    if total_seqlen <= chunk_size:
        return fn(*args, **kwargs)

    results: list = []
    for i in range(0, total_seqlen, chunk_size):
        end_idx = min(i + chunk_size, total_seqlen)
        # Slice tensors that match the total sequence length; pass others as-is
        chunk_args = [
            arg[i:end_idx] if isinstance(arg, torch.Tensor) and arg.shape[0] == total_seqlen else arg 
            for arg in args
        ]
        results.append(fn(*chunk_args, **kwargs))

    # Handle single tensor vs tuple of tensors output
    if isinstance(results[0], tuple):
        num_outputs = len(results[0])
        return tuple(torch.cat([r[i] for r in results]) for i in range(num_outputs))
    return torch.cat(results)


# =============================================================================
# Vocab Parallel Autograd Functions
# =============================================================================


class _VocabParallelLogProbs(torch.autograd.Function):
    """Compute log probabilities when logits are sharded on the vocab dimension."""

    @staticmethod
    def forward(
        ctx,
        vocab_parallel_logits: torch.Tensor,
        labels: torch.Tensor,
        tp_group: dist.ProcessGroup,
    ) -> torch.Tensor:
        tp_rank = dist.get_rank(tp_group)
        partition_vocab_size = vocab_parallel_logits.size(-1)
        vocab_start_index = tp_rank * partition_vocab_size
        vocab_end_index = vocab_start_index + partition_vocab_size

        # Numerical stability - subtract max
        logits_max = vocab_parallel_logits.max(dim=-1, keepdim=True).values
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=tp_group)
        normalized_logits = vocab_parallel_logits - logits_max

        # Compute exp and sum
        exp_logits = normalized_logits.exp()
        sum_exp_logits = exp_logits.sum(dim=-1, keepdim=True)
        dist.all_reduce(sum_exp_logits, op=dist.ReduceOp.SUM, group=tp_group)

        # Get the logit value corresponding to the label
        # Create a mask for labels that fall into the local vocab partition
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

        # Compute log probability: log(softmax) = logits - log(sum(exp))
        log_sum_exp = sum_exp_logits.squeeze(-1).log()
        logprobs = predicted_logits - log_sum_exp

        # Compute softmax in-place for backward (re-use exp_logits memory)
        softmax = exp_logits.div_(sum_exp_logits)
        ctx.save_for_backward(softmax, labels_mask, masked_labels_1d)

        return logprobs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        softmax, labels_mask, masked_labels_1d = ctx.saved_tensors
        partition_vocab_size = softmax.size(-1)

        # Grad input initialization (view as 2D for easier indexing)
        grad_input = softmax
        grad_2d = grad_input.view(-1, partition_vocab_size)
        arange_1d = torch.arange(grad_2d.size(0), device=grad_2d.device)

        # Update gradient at label positions: grad = softmax - 1 (since grad_output propagates -1 * impact)
        # Actually standard softmax grad is p_i - y_i.
        update_mask = ~labels_mask.view(-1)
        grad_2d[arange_1d, masked_labels_1d] -= update_mask.float()

        # Chain rule: multiply by incoming gradient
        grad_input.mul_(grad_output.unsqueeze(-1))
        
        grad_input.neg_()

        return grad_input, None, None


class _VocabParallelEntropy(torch.autograd.Function):
    """Compute entropy when logits are sharded on the vocab dimension."""

    @staticmethod
    def forward(ctx, vocab_parallel_logits: torch.Tensor, tp_group: dist.ProcessGroup) -> torch.Tensor:
        # Max for stability
        logits_max = vocab_parallel_logits.max(dim=-1, keepdim=True).values
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=tp_group)

        # Softmax components
        normalized_logits = vocab_parallel_logits - logits_max
        exp_logits = normalized_logits.exp()
        sum_exp_logits = exp_logits.sum(dim=-1, keepdim=True)
        dist.all_reduce(sum_exp_logits, op=dist.ReduceOp.SUM, group=tp_group)

        softmax = exp_logits.div_(sum_exp_logits)
        
        # Calculate sum(p * x) distributedly
        sum_softmax_times_logits = (softmax * vocab_parallel_logits).sum(dim=-1, keepdim=True)
        dist.all_reduce(sum_softmax_times_logits, op=dist.ReduceOp.SUM, group=tp_group)

        # Entropy = log(Z) - E[x]
        entropy = (logits_max + sum_exp_logits.log() - sum_softmax_times_logits).squeeze(-1)
        
        ctx.save_for_backward(vocab_parallel_logits, softmax, sum_softmax_times_logits)
        return entropy

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        vocab_parallel_logits, softmax, sum_softmax_times_logits = ctx.saved_tensors
        grad_input = softmax * (vocab_parallel_logits - sum_softmax_times_logits)
        grad_input.mul_(grad_output.unsqueeze(-1))
        grad_input.neg_()
        return grad_input, None


class _VocabParallelCrossEntropy(torch.autograd.Function):
    """Compute cross entropy loss when logits are sharded on the vocab dimension."""

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

        ignore_mask = labels == ignore_index

        # Stability
        logits_max = vocab_parallel_logits.max(dim=-1, keepdim=True).values
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=tp_group)
        normalized_logits = vocab_parallel_logits - logits_max

        # Exp & Sum
        exp_logits = normalized_logits.exp()
        sum_exp_logits = exp_logits.sum(dim=-1, keepdim=True)
        dist.all_reduce(sum_exp_logits, op=dist.ReduceOp.SUM, group=tp_group)

        # Locate target logits
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

        # Loss = log(sum(exp)) - target_logit
        log_sum_exp = sum_exp_logits.squeeze(-1).log()
        loss = log_sum_exp - predicted_logits
        loss[ignore_mask] = 0.0

        softmax = exp_logits.div_(sum_exp_logits)
        ctx.save_for_backward(softmax, labels_mask, masked_labels_1d, ignore_mask)

        return loss

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        softmax, labels_mask, masked_labels_1d, ignore_mask = ctx.saved_tensors
        partition_vocab_size = softmax.size(-1)

        grad_input = softmax.clone()
        grad_2d = grad_input.view(-1, partition_vocab_size)
        arange_1d = torch.arange(grad_2d.size(0), device=grad_2d.device)

        # Gradient: p - y (where y is one-hot target)
        update_mask = ~labels_mask.view(-1)
        grad_2d[arange_1d, masked_labels_1d] -= update_mask.float()
        
        grad_input.mul_(grad_output.unsqueeze(-1))

        if ignore_mask.any():
            grad_input[ignore_mask.view(*ignore_mask.shape, 1).expand_as(grad_input)] = 0.0

        return grad_input, None, None, None


# =============================================================================
# Public API (TP only)
# =============================================================================


def vocab_parallel_logprobs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    tp_group: dist.ProcessGroup,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute log probabilities for vocab-sharded logits (TP > 1)."""
    if temperature != 1.0:
        logits = logits.float() / temperature
    else:
        logits = logits.float()

    batch_shape = logits.shape[:-1]
    vocab_size = logits.shape[-1]
    
    # Flatten for chunking
    logits_2d = logits.reshape(-1, vocab_size)
    labels_1d = labels.reshape(-1)

    result = _chunked_apply(
        _VocabParallelLogProbs.apply, 
        logits_2d, 
        labels_1d, 
        chunk_size=1024, 
        tp_group=tp_group
    )
    return result.view(*batch_shape)


def vocab_parallel_entropy(
    logits: torch.Tensor,
    tp_group: dist.ProcessGroup,
) -> torch.Tensor:
    """Compute entropy for vocab-sharded logits (TP > 1)."""
    logits = logits.float()
    batch_shape = logits.shape[:-1]
    vocab_size = logits.shape[-1]
    
    logits_2d = logits.reshape(-1, vocab_size)

    result = _chunked_apply(
        _VocabParallelEntropy.apply, 
        logits_2d, 
        chunk_size=1024, 
        tp_group=tp_group
    )
    return result.view(*batch_shape)


def vocab_parallel_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    tp_group: dist.ProcessGroup,
    ignore_index: int = -100,
    reduction: str = "none",
) -> torch.Tensor:
    """Compute cross entropy loss for vocab-sharded logits (TP > 1)."""
    logits = logits.float()
    batch_shape = logits.shape[:-1]
    vocab_size = logits.shape[-1]
    
    logits_2d = logits.reshape(-1, vocab_size)
    labels_1d = labels.reshape(-1)

    loss = _chunked_apply(
        _VocabParallelCrossEntropy.apply, 
        logits_2d, 
        labels_1d, 
        chunk_size=1024, 
        tp_group=tp_group, 
        ignore_index=ignore_index
    )
    loss = loss.view(*batch_shape)

    if reduction == "none":
        return loss
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "mean":
        valid_tokens = (labels != ignore_index).sum()
        return loss.sum() / valid_tokens.clamp(min=1)
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


def vocab_parallel_logprobs_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    tp_group: dist.ProcessGroup,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute both logprobs and entropy for vocab-sharded logits."""
    return (
        vocab_parallel_logprobs(logits, labels, tp_group=tp_group, temperature=temperature),
        vocab_parallel_entropy(logits if temperature == 1.0 else (logits.float() / temperature), tp_group=tp_group),
    )


# =============================================================================
# Unified API (DTensor-aware) - Recommended
# =============================================================================


def _is_dtensor(x: object) -> bool:
    return DTensor is not None and isinstance(x, DTensor)


def _dtensor_is_vocab_sharded(logits: "DTensor") -> bool:
    if Shard is None:
        return False
    # Check if any placement is sharded on the last dimension (vocab)
    for p in logits.placements:
        if isinstance(p, Shard) and p.dim in (-1, logits.ndim - 1):
            return True
    return False


def prepare_vocab_parallel_logits(
    logits: torch.Tensor,
    tp_group: Optional[dist.ProcessGroup] = None,
) -> Tuple[torch.Tensor, Optional[dist.ProcessGroup]]:
    """
    Ensure logits are ready for vocab-parallel processing.
    - If DTensor: extract local shard and auto-detect TP group if V-sharded.
    - If Tensor: return as-is.
    """
    if _is_dtensor(logits):
        assert DTensor is not None
        logits_dt: DTensor = logits  # type: ignore
        local = logits_dt.to_local()
        # Auto-detect TP group if the DTensor is sharded on vocab dim
        if tp_group is None and _dtensor_is_vocab_sharded(logits_dt):
            tp_group = logits_dt.device_mesh.get_group()
        return local, tp_group
    return logits, tp_group


def log_probs_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    tp_group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """
    Compute log probabilities.
    Automatically handles both vocab-parallel sharded logits (via chunking) 
    and standard full logits (via chunking) for memory efficiency.
    """
    local_logits, tp_group = prepare_vocab_parallel_logits(logits, tp_group=tp_group)

    if tp_group is not None and dist.get_world_size(tp_group) > 1:
        return vocab_parallel_logprobs(local_logits, labels, tp_group=tp_group, temperature=temperature)

    # Helper for chunked execution on non-TP/single-device logits
    def _compute_local_logprobs(l_logits, l_labels):
        scaled = l_logits.float() / temperature if temperature != 1.0 else l_logits.float()
        log_probs = torch.log_softmax(scaled, dim=-1)
        return log_probs.gather(dim=-1, index=l_labels.unsqueeze(-1)).squeeze(-1)

    batch_shape = local_logits.shape[:-1]
    vocab_size = local_logits.shape[-1]
    logits_2d = local_logits.reshape(-1, vocab_size)
    labels_1d = labels.reshape(-1)

    result = _chunked_apply(_compute_local_logprobs, logits_2d, labels_1d, chunk_size=1024)
    return result.view(*batch_shape)


def compute_entropy(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    tp_group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """Compute entropy with memory-efficient chunking."""
    local_logits, tp_group = prepare_vocab_parallel_logits(logits, tp_group=tp_group)

    if tp_group is not None and dist.get_world_size(tp_group) > 1:
        scaled = local_logits.float() / temperature if temperature != 1.0 else local_logits.float()
        return vocab_parallel_entropy(scaled, tp_group=tp_group)

    def _compute_local_entropy(l_logits):
        scaled = l_logits.float() / temperature if temperature != 1.0 else l_logits.float()
        log_z = torch.logsumexp(scaled, dim=-1)
        p = torch.softmax(scaled, dim=-1)
        return log_z - (p * scaled).sum(dim=-1)
    
    batch_shape = local_logits.shape[:-1]
    vocab_size = local_logits.shape[-1]
    logits_2d = local_logits.reshape(-1, vocab_size)

    result = _chunked_apply(_compute_local_entropy, logits_2d, chunk_size=1024)
    return result.view(*batch_shape)


def cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
    reduction: str = "none",
    tp_group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """
    Compute cross entropy loss.
    Supports vocab parallelism (TP) and chunked execution for memory efficiency.
    """
    local_logits, tp_group = prepare_vocab_parallel_logits(logits, tp_group=tp_group)

    if tp_group is not None and dist.get_world_size(tp_group) > 1:
        return vocab_parallel_cross_entropy(
            local_logits,
            labels,
            tp_group=tp_group,
            ignore_index=ignore_index,
            reduction=reduction,
        )

    vocab = local_logits.size(-1)
    logits_2d = local_logits.reshape(-1, vocab)
    labels_1d = labels.reshape(-1)
    
    def _compute_local_ce(l, t):
        return F.cross_entropy(l.float(), t, ignore_index=ignore_index, reduction="none")
    
    loss = _chunked_apply(_compute_local_ce, logits_2d, labels_1d, chunk_size=1024).view_as(labels)

    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        valid = (labels != ignore_index).sum()
        return loss.sum() / valid.clamp(min=1)
    raise ValueError(f"Invalid reduction: {reduction}")


def _vocab_parallel_argmax(
    vocab_parallel_logits: torch.Tensor,
    tp_group: dist.ProcessGroup,
) -> torch.Tensor:
    """Compute global argmax from vocab-sharded logits."""
    tp_size = dist.get_world_size(tp_group)
    if tp_size == 1:
        return vocab_parallel_logits.argmax(dim=-1)

    tp_rank = dist.get_rank(tp_group)
    shard_vocab = vocab_parallel_logits.size(-1)
    flat = vocab_parallel_logits.reshape(-1, shard_vocab)
    local_max, local_argmax = flat.max(dim=-1)
    # Convert local index to global index
    local_argmax = local_argmax + tp_rank * shard_vocab

    # Gather results from all ranks
    max_list = [torch.empty_like(local_max) for _ in range(tp_size)]
    idx_list = [torch.empty_like(local_argmax) for _ in range(tp_size)]
    dist.all_gather(max_list, local_max, group=tp_group)
    dist.all_gather(idx_list, local_argmax, group=tp_group)

    max_stack = torch.stack(max_list, dim=0)
    idx_stack = torch.stack(idx_list, dim=0)
    
    # Find the winner across ranks
    winner = max_stack.argmax(dim=0)
    pred = idx_stack.gather(dim=0, index=winner.unsqueeze(0)).squeeze(0)
    return pred.view(vocab_parallel_logits.shape[:-1])


def cross_entropy_loss_with_acc(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
    reduction: str = "mean",
    tp_group: Optional[dist.ProcessGroup] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute cross entropy loss and accuracy."""
    local_logits, tp_group = prepare_vocab_parallel_logits(logits, tp_group=tp_group)

    if tp_group is not None and dist.get_world_size(tp_group) > 1:
        loss = vocab_parallel_cross_entropy(
            local_logits, labels, tp_group=tp_group, ignore_index=ignore_index, reduction=reduction
        )
        pred = _vocab_parallel_argmax(local_logits, tp_group=tp_group)
    else:
        loss = cross_entropy_loss(
            local_logits, labels, ignore_index=ignore_index, reduction=reduction, tp_group=None
        )
        pred = local_logits.argmax(dim=-1)

    valid = labels != ignore_index
    correct = (pred == labels) & valid
    acc = correct.float().sum() / valid.sum().clamp(min=1)
    return loss, acc


def select_token_logits(
    logits: torch.Tensor,
    token_ids: Union[List[int], Tuple[int, ...]],
    *,
    tp_group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """
    Select logits for specific token IDs.
    Handles vocab-parallelism by reducing results across ranks.
    """
    if not token_ids:
        raise ValueError("token_ids must be non-empty")

    local_logits, tp_group = prepare_vocab_parallel_logits(logits, tp_group=tp_group)
    token_ids_t = torch.as_tensor(token_ids, device=local_logits.device, dtype=torch.long)

    if tp_group is None or dist.get_world_size(tp_group) == 1:
        return local_logits.index_select(dim=-1, index=token_ids_t)

    # TP logic: check which tokens belong to this shard
    tp_rank = dist.get_rank(tp_group)
    shard_vocab = local_logits.size(-1)
    vocab_start = tp_rank * shard_vocab
    vocab_end = vocab_start + shard_vocab

    in_range = (token_ids_t >= vocab_start) & (token_ids_t < vocab_end)
    local_idx = (token_ids_t - vocab_start).clamp(min=0, max=shard_vocab - 1)
    
    selected = local_logits.index_select(dim=-1, index=local_idx)
    # Zero out selections that aren't in this shard
    selected = selected * in_range.to(dtype=selected.dtype)
    
    # Sum up across ranks to get the full values
    dist.all_reduce(selected, op=dist.ReduceOp.SUM, group=tp_group)
    return selected
