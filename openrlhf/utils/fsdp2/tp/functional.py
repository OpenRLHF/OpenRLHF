"""
Unified TP-aware Functional API
================================

This module provides a unified API that automatically detects Tensor Parallelism
and routes to the appropriate implementation:
- TP enabled (vocab sharded): Use vocab_parallel implementations
- TP disabled (regular tensor): Use standard PyTorch implementations

Usage:
    from openrlhf.utils.fsdp2.tp.functional import log_probs_from_logits, cross_entropy_loss
    
    # Automatically handles both TP and non-TP cases
    log_probs = log_probs_from_logits(logits, labels)
    loss = cross_entropy_loss(logits, labels)
"""

import torch
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional, Tuple


def _is_vocab_sharded(logits: torch.Tensor) -> Tuple[bool, Optional[object]]:
    """Check if logits are sharded on vocab dimension (TP enabled).
    
    Returns:
        (is_sharded, tp_group): Whether vocab is sharded and the TP process group
    """
    try:
        from torch.distributed.tensor import DTensor
        from torch.distributed.tensor.placement_types import Shard
    except ImportError:
        return False, None
    
    if not isinstance(logits, DTensor):
        return False, None
    
    placements = getattr(logits, "placements", ())
    is_sharded = any(
        isinstance(p, Shard) and getattr(p, "dim", None) in (-1, logits.ndim - 1) 
        for p in placements
    )
    
    if is_sharded:
        tp_group = logits.device_mesh.get_group()
        return True, tp_group
    
    return False, None


def _to_local_if_dtensor(logits: torch.Tensor) -> torch.Tensor:
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        return logits

    return logits.to_local() if isinstance(logits, DTensor) else logits


def _logsumexp_by_chunk(logits: torch.Tensor, chunk_size: int = 1024) -> torch.Tensor:
    """Chunked logsumexp for memory efficiency."""
    seq_len = logits.shape[0]
    logsumexp_values = torch.zeros((seq_len,), device=logits.device, dtype=logits.dtype)
    for s_idx in range(0, seq_len, chunk_size):
        end_idx = min(s_idx + chunk_size, seq_len)
        logsumexp_values[s_idx:end_idx] = torch.logsumexp(logits[s_idx:end_idx], dim=-1)
    return logsumexp_values


def prepare_vocab_parallel_logits(
    logits: torch.Tensor,
) -> Tuple[torch.Tensor, Optional[dist.ProcessGroup]]:
    """Return local logits and TP group if vocab is sharded."""
    is_sharded, tp_group = _is_vocab_sharded(logits)
    return _to_local_if_dtensor(logits), tp_group if is_sharded else None


def select_token_logits(
    logits: torch.Tensor,
    token_ids: list[int] | torch.Tensor,
    tp_group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """Select logits for token IDs, handling vocab-sharded TP if provided."""
    if tp_group is None:
        is_sharded, tp_group = _is_vocab_sharded(logits)
    else:
        is_sharded = True

    local_logits = _to_local_if_dtensor(logits)
    if not is_sharded or tp_group is None or dist.get_world_size(tp_group) == 1:
        return local_logits[..., token_ids]

    if isinstance(token_ids, torch.Tensor):
        token_ids_list = token_ids.detach().cpu().tolist()
    else:
        token_ids_list = [int(token) for token in token_ids]

    tp_rank = dist.get_rank(tp_group)
    partition_vocab_size = local_logits.size(-1)
    vocab_start = tp_rank * partition_vocab_size
    vocab_end = vocab_start + partition_vocab_size

    out = []
    for token in token_ids_list:
        token_mask = (token < vocab_start) or (token >= vocab_end)
        local_idx = token - vocab_start if not token_mask else 0
        token_logits = local_logits[..., local_idx].clone()
        if token_mask:
            token_logits.zero_()
        dist.all_reduce(token_logits, op=dist.ReduceOp.SUM, group=tp_group)
        out.append(token_logits)
    return torch.stack(out, dim=-1)


def log_probs_from_logits(
    logits: torch.Tensor, 
    labels: torch.Tensor, 
    temperature: float = 1.0
) -> torch.Tensor:
    """Compute log probabilities from logits.
    
    Automatically handles:
    - TP mode: Uses vocab_parallel_logprobs when logits are sharded
    - Non-TP mode: Uses standard PyTorch implementation
    
    Args:
        logits: [..., vocab_size] logits tensor (may be DTensor with Shard(-1))
        labels: [...] label indices
        temperature: Temperature for softmax
        
    Returns:
        [...] log probabilities for each label
    """
    is_sharded, tp_group = _is_vocab_sharded(logits)
    
    if is_sharded:
        from .vocab_parallel import vocab_parallel_logprobs
        
        local_logits = logits if temperature == 1.0 else logits / float(temperature)
        return vocab_parallel_logprobs(local_logits, labels, tp_group=tp_group, temperature=1.0)
    
    # Non-TP path: convert DTensor to local if needed
    try:
        from torch.distributed.tensor import DTensor
        if isinstance(logits, DTensor):
            logits = logits.to_local()
    except ImportError:
        pass
    
    if temperature != 1.0:
        logits = logits / float(temperature)
    
    # Optimized path for fp32/fp64
    if logits.dtype in [torch.float32, torch.float64]:
        batch_dim = logits.shape[:-1]
        last_dim = logits.shape[-1]
        try:
            from flash_attn.ops.triton.cross_entropy import cross_entropy_loss
            output = cross_entropy_loss(logits.reshape(-1, last_dim), labels.reshape(-1))
            return -output[0].view(*batch_dim)
        except ImportError:
            logits_labels = torch.gather(logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
            logsumexp_values = _logsumexp_by_chunk(logits.reshape(-1, last_dim))
            logsumexp_values = logsumexp_values.view(*batch_dim)
            return logits_labels - logsumexp_values
    
    # Default path: row-by-row to reduce memory
    log_probs_labels = []
    for row_logits, row_labels in zip(logits, labels):
        row_log_probs = F.log_softmax(row_logits, dim=-1)
        row_log_probs_labels = row_log_probs.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
        log_probs_labels.append(row_log_probs_labels)
    return torch.stack(log_probs_labels)


def cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = "mean",
    tp_group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """Compute cross entropy loss.
    
    Automatically handles:
    - TP mode: Uses vocab_parallel_cross_entropy when logits are sharded
    - Non-TP mode: Uses standard F.cross_entropy
    
    Args:
        logits: [..., vocab_size] logits tensor (may be DTensor with Shard(-1))
        labels: [...] label indices
        ignore_index: Index to ignore in loss computation
        reduction: 'mean', 'sum', or 'none'
        tp_group: TP process group when logits are already local shards
        
    Returns:
        Cross entropy loss
    """
    if tp_group is None:
        is_sharded, tp_group = _is_vocab_sharded(logits)
    else:
        is_sharded = True
    
    if is_sharded:
        from .vocab_parallel import vocab_parallel_cross_entropy
        return vocab_parallel_cross_entropy(
            logits, labels, 
            tp_group=tp_group, 
            ignore_index=ignore_index, 
            reduction=reduction
        )
    
    # Non-TP path: convert DTensor to local if needed
    logits = _to_local_if_dtensor(logits)
    
    return F.cross_entropy(logits, labels, ignore_index=ignore_index, reduction=reduction)


def cross_entropy_loss_with_acc(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = "mean",
    tp_group: Optional[dist.ProcessGroup] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute cross entropy loss and accuracy.
    
    Automatically handles TP mode for both loss and accuracy computation.
    
    Args:
        logits: [..., vocab_size] logits tensor (may be DTensor with Shard(-1))
        labels: [...] label indices
        ignore_index: Index to ignore in loss computation
        reduction: 'mean', 'sum', or 'none'
        tp_group: TP process group when logits are already local shards
        
    Returns:
        (loss, accuracy) tuple
    """
    if tp_group is None:
        is_sharded, tp_group = _is_vocab_sharded(logits)
    else:
        is_sharded = True
    
    if is_sharded:
        from .vocab_parallel import vocab_parallel_cross_entropy
        
        loss = vocab_parallel_cross_entropy(
            logits, labels, 
            tp_group=tp_group, 
            ignore_index=ignore_index, 
            reduction=reduction
        )
        
        # Compute accuracy with sharded logits
        local_logits = _to_local_if_dtensor(logits)
        tp_world = dist.get_world_size(tp_group)
        local_max, local_idx = local_logits.max(dim=-1)
        max_list = [torch.empty_like(local_max) for _ in range(tp_world)]
        idx_list = [torch.empty_like(local_idx) for _ in range(tp_world)]
        dist.all_gather(max_list, local_max, group=tp_group)
        dist.all_gather(idx_list, local_idx, group=tp_group)
        
        max_stack = torch.stack(max_list, dim=0)
        idx_stack = torch.stack(idx_list, dim=0)
        best_rank = max_stack.argmax(dim=0)
        best_local = idx_stack.gather(0, best_rank.unsqueeze(0)).squeeze(0)
        pred = best_rank * local_logits.size(-1) + best_local
        acc = (pred == labels).float().mean()
        
        return loss, acc
    
    # Non-TP path
    logits = _to_local_if_dtensor(logits)
    
    loss = F.cross_entropy(logits, labels, ignore_index=ignore_index, reduction=reduction)
    acc = (logits.argmax(dim=-1) == labels).float().mean()
    
    return loss, acc
