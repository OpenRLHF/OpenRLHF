"""Loss-parallel utilities for vocab-sharded DTensor logits.

When TP (Tensor Parallelism) is enabled, the LM head's vocab dimension is sharded
across ranks. This module provides sharded versions of common loss/metric operations
so they can work directly on sharded logits without gathering the full vocab.
"""

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
from torch.distributed.tensor import DTensor


def _allreduce_sum(x: torch.Tensor, group, *, mean_grad: bool = False) -> torch.Tensor:
    """All-reduce SUM with selectable backward semantics."""
    if not dist.is_initialized():
        return x

    if not mean_grad:
        return dist_nn.all_reduce(x, op=dist.ReduceOp.SUM, group=group)

    class _AllReduceSumMeanGrad(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor, group) -> torch.Tensor:  # type: ignore[override]
            ctx.group = group
            ctx.group_size = dist.get_world_size(group=group)
            out = x.clone()
            dist.all_reduce(out, op=dist.ReduceOp.SUM, group=group)
            return out

        @staticmethod
        def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
            grad = grad_out.clone()
            dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=ctx.group)
            grad.div_(float(ctx.group_size))
            return grad, None

    return _AllReduceSumMeanGrad.apply(x, group)


def _unpack_sharded_logits(tensor: DTensor):
    """Extract sharding info from a vocab-sharded DTensor."""
    mesh = tensor.device_mesh
    group = mesh.get_group()
    rank = dist.get_rank(group)
    world = dist.get_world_size(group)
    local_logits = tensor.to_local()
    local_vocab_size = local_logits.size(-1)

    # All-gather each rank's local vocab size — shards may differ when vocab_size % world != 0
    size_tensor = torch.tensor(local_vocab_size, device=local_logits.device, dtype=torch.int64)
    size_list = [torch.zeros_like(size_tensor) for _ in range(world)]
    dist.all_gather(size_list, size_tensor, group=group)
    sizes = [int(s.item()) for s in size_list]
    # Compute this rank's global vocab index range: [vocab_start, vocab_end)
    vocab_start = sum(sizes[:rank])
    vocab_end = vocab_start + sizes[rank]
    global_vocab_size = sum(sizes)
    return group, local_logits, local_vocab_size, global_vocab_size, vocab_start, vocab_end


def _sharded_logsumexp(
    local_logits: torch.Tensor,
    group,
    dim: int = -1,
    *,
    mean_grad: bool = False,
) -> torch.Tensor:
    """Compute logsumexp over sharded vocab dimension.

    Uses the max-shift trick for numerical stability:
        logsumexp(x) = max(x) + log(sum(exp(x - max(x))))
    """
    # Step 1: find the global max across all ranks (for numerical stability)
    local_max = local_logits.max(dim=dim).values
    global_max = local_max.detach().clone()
    dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=group)
    # Step 2: each rank computes exp(x_local - global_max) to avoid overflow, then sums locally
    local_exp = torch.exp(local_logits - global_max.unsqueeze(dim))
    local_exp_sum = local_exp.sum(dim=dim)
    # Step 3: allreduce the partial exp sums to get the global sum
    global_exp_sum = _allreduce_sum(local_exp_sum, group, mean_grad=mean_grad)
    # Reassemble: logsumexp = global_max + log(global_exp_sum)
    return global_max + torch.log(global_exp_sum)


def compute_token_log_probs_sharded(
    logits: DTensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Sharded version of ``_compute_token_log_probs_local`` (openrlhf.models.utils).

    Core formula: log_prob(label) = logit[label] - logsumexp(logits)
    """
    group, local_logits, local_vocab_size, _, vocab_start, vocab_end = _unpack_sharded_logits(logits)
    local_logits_f32 = local_logits.float()
    if temperature != 1.0:
        local_logits_f32 = local_logits_f32 / temperature

    # logsumexp over the full (global) vocab, computed in sharded fashion
    logsumexp_global = _sharded_logsumexp(local_logits_f32, group, dim=-1, mean_grad=True)

    labels = labels.to(local_logits_f32.device)
    local_label_indices = labels - vocab_start
    # label_in_shard: True only if this rank's vocab shard contains the target label
    label_in_shard = (labels >= vocab_start) & (labels < vocab_end) & (labels != ignore_index)
    # clamped_local_idx: clamped index so gather() won't OOB — the value is discarded when label_in_shard=False
    clamped_local_idx = local_label_indices.clamp(0, max(local_vocab_size - 1, 0))
    gathered_logits = local_logits_f32.gather(dim=-1, index=clamped_local_idx.unsqueeze(-1)).squeeze(-1)
    # Zero out contributions from ranks that don't own the label's vocab entry
    local_label_logits = torch.where(label_in_shard, gathered_logits, torch.zeros_like(gathered_logits))

    # Allreduce: exactly one rank contributes the real logit value; others contribute zero
    global_label_logits = _allreduce_sum(local_label_logits, group, mean_grad=True)

    # log_prob = logit[label] - logsumexp(logits)
    log_probs = global_label_logits - logsumexp_global
    log_probs = torch.where(labels == ignore_index, torch.zeros_like(log_probs), log_probs)
    return log_probs


def compute_entropy_sharded(logits: DTensor) -> torch.Tensor:
    """Sharded version of ``_compute_entropy_local`` (openrlhf.models.utils).

    Entropy: H = -sum(p * log(p)), computed as:
        1. max-shift for numerical stability (same as _sharded_logsumexp)
        2. local softmax → allreduce normalizer → local p*log(p) → allreduce sum
    """

    class _ShardedEntropyFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, local_logits: torch.Tensor, group) -> torch.Tensor:  # type: ignore[override]
            dim = -1
            # --- max-shift trick (same idea as _sharded_logsumexp) ---
            local_max = local_logits.max(dim=dim).values
            global_max = local_max.detach().clone()
            dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=group)

            # logits_shifted = x - max(x), so exp(logits_shifted) won't overflow
            logits_shifted = local_logits - global_max.unsqueeze(dim)
            local_exp = torch.exp(logits_shifted)
            local_exp_sum = local_exp.sum(dim=dim)

            # Allreduce to get the global partition function Z = sum(exp(logits_shifted))
            global_exp_sum = local_exp_sum.detach().clone()
            dist.all_reduce(global_exp_sum, op=dist.ReduceOp.SUM, group=group)

            # p_i = exp(logits_shifted_i) / Z  (local softmax using global normalizer)
            local_probs = local_exp / global_exp_sum.unsqueeze(dim)
            # log(p_i) = logits_shifted_i - log(Z)
            local_log_probs = logits_shifted - torch.log(global_exp_sum).unsqueeze(dim)

            # sum(p * log(p)) — each rank computes its local vocab slice, then allreduce
            local_plogp_sum = (local_probs * local_log_probs).sum(dim=dim)
            global_plogp_sum = local_plogp_sum.detach().clone()
            dist.all_reduce(global_plogp_sum, op=dist.ReduceOp.SUM, group=group)
            entropy = -global_plogp_sum

            ctx.save_for_backward(local_probs, local_log_probs, entropy)
            return entropy

        @staticmethod
        def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
            local_probs, local_log_probs, entropy = ctx.saved_tensors
            # dH/dx_i = -p_i * (H + log(p_i)), derived from d/dx[-sum(p*log(p))]
            grad_local_logits = grad_out.unsqueeze(-1) * (-local_probs * (entropy.unsqueeze(-1) + local_log_probs))
            return grad_local_logits, None

    group, local_logits, _, _, _, _ = _unpack_sharded_logits(logits)
    return _ShardedEntropyFn.apply(local_logits.float(), group)


def gather_token_logits_sharded(logits: DTensor, token_ids: torch.Tensor) -> torch.Tensor:
    """Sharded version of ``torch.gather``/``index_select`` for vocab-sharded DTensor logits.

    Same pattern as the label-gather in compute_token_log_probs_sharded:
    check if token_id is in the local shard → safe index → zero out non-owners → allreduce.
    """
    group, local_logits, _, _, vocab_start, vocab_end = _unpack_sharded_logits(logits)
    token_ids = token_ids.to(local_logits.device)
    local_token_indices = token_ids - vocab_start
    # token_in_shard: True if this rank's vocab shard contains the requested token_id
    token_in_shard = (token_ids >= vocab_start) & (token_ids < vocab_end)
    # clamped_local_idx: clamped to valid range; result discarded when token_in_shard=False
    clamped_local_idx = local_token_indices.clamp(0, max=local_logits.size(-1) - 1)
    gathered_logits = local_logits.index_select(dim=-1, index=clamped_local_idx)
    # Only the owning rank keeps its gathered value; others contribute zero
    local_masked_logits = torch.where(token_in_shard, gathered_logits, torch.zeros_like(gathered_logits))
    return _allreduce_sum(local_masked_logits, group, mean_grad=True)


def compute_kd_loss_sharded(
    logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    label: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Sharded version of ``KDLoss.forward`` (openrlhf.models.loss).

    Core formula: KD loss = -sum(teacher_prob * student_log_prob)
    Teacher and student may independently be DTensor or plain Tensor,
    so logsumexp is computed with the sharded or local path accordingly.
    """

    def _get_local_logits(logits: torch.Tensor, vocab_start: int, vocab_end: int) -> torch.Tensor:
        """Get local vocab slice from logits (handles both DTensor and regular Tensor)."""
        if isinstance(logits, DTensor):
            return logits.to_local()
        return logits[..., vocab_start:vocab_end]

    # Use whichever tensor is a DTensor to derive the sharding metadata
    dtensor_ref = logits if isinstance(logits, DTensor) else teacher_logits
    if not isinstance(dtensor_ref, DTensor):
        raise TypeError("compute_kd_loss_sharded requires logits or teacher_logits to be DTensor")

    group, _, _, _, vocab_start, vocab_end = _unpack_sharded_logits(dtensor_ref)

    local_student_logits = _get_local_logits(logits, vocab_start, vocab_end).float()
    local_teacher_logits = _get_local_logits(teacher_logits, vocab_start, vocab_end).float()

    # logsumexp: use sharded path if the tensor is DTensor, otherwise local-only is correct
    if isinstance(teacher_logits, DTensor):
        teacher_logsumexp = _sharded_logsumexp(local_teacher_logits, group, dim=-1, mean_grad=False)
    else:
        teacher_logsumexp = torch.logsumexp(teacher_logits.float(), dim=-1)

    if isinstance(logits, DTensor):
        student_logsumexp = _sharded_logsumexp(local_student_logits, group, dim=-1, mean_grad=False)
    else:
        student_logsumexp = torch.logsumexp(logits.float(), dim=-1)

    # teacher_prob * student_log_prob, summed over the local vocab slice
    local_teacher_probs = torch.exp(local_teacher_logits - teacher_logsumexp.unsqueeze(-1))
    local_student_logprobs = local_student_logits - student_logsumexp.unsqueeze(-1)
    local_cross_entropy = (local_teacher_probs * local_student_logprobs).sum(dim=-1)

    # Allreduce partial sums across ranks to get the global cross-entropy
    global_cross_entropy = _allreduce_sum(local_cross_entropy, group, mean_grad=True)

    mask = (label != ignore_index).float()
    # KD loss = -cross_entropy, averaged over non-ignored tokens
    return -(global_cross_entropy.view(-1) * mask.view(-1)).sum() / mask.sum()


def compute_argmax_sharded(logits: DTensor, mask: torch.Tensor) -> torch.Tensor:
    """Sharded version of ``torch.argmax`` for vocab-sharded DTensor logits.

    Two-step allreduce:
        1. Find the global max *value* across all ranks.
        2. The rank holding that max converts its local argmax to a global index;
           others contribute -1. A second allreduce(MAX) picks the global index.
    """
    group, local_logits, _, _, vocab_start, _ = _unpack_sharded_logits(logits)
    masked_local_logits = local_logits.detach()[mask].float()
    local_max, local_argmax = masked_local_logits.max(dim=-1)

    # Step 1: find the global maximum value
    global_max = local_max.clone()
    dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=group)

    # Step 2: only the rank whose local_max equals global_max contributes
    # its global index (local_argmax + vocab_start); others emit -1
    candidate_idx = torch.where(
        local_max == global_max,
        local_argmax + vocab_start,
        torch.full_like(local_argmax, -1),
    )
    # allreduce(MAX) selects the valid global index (>= 0) from the winning rank
    global_argmax = candidate_idx.clone()
    dist.all_reduce(global_argmax, op=dist.ReduceOp.MAX, group=group)
    return global_argmax
