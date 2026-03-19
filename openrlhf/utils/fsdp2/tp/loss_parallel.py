"""Loss-parallel utilities for vocab-sharded DTensor logits.

When TP shards the LM head's vocab dimension across ranks, these functions
compute loss/metrics directly on sharded logits without gathering the full vocab.

Key pattern: each rank holds logits[..., vocab_start:vocab_end]. Shared operations
(logsumexp, argmax) use all-reduce across the TP group.
"""

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
from torch.distributed.tensor import DTensor


# =============================================================================
# Primitives
# =============================================================================


def _allreduce_sum(x: torch.Tensor, group, *, mean_grad: bool = False) -> torch.Tensor:
    """All-reduce SUM in forward. If mean_grad=True, backward uses SUM/world_size.

    mean_grad=True is needed when a loss is computed from sharded pieces:
    forward aggregates via SUM, but each rank's gradient should be averaged
    (not duplicated) to get the correct global gradient.
    """
    if not dist.is_initialized():
        return x

    if not mean_grad:
        # Standard: forward=SUM, backward=SUM
        return dist_nn.all_reduce(x, op=dist.ReduceOp.SUM, group=group)

    # Custom: forward=SUM, backward=SUM/N (mean gradient)
    class _AllReduceSumMeanGrad(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, group):
            ctx.group = group
            ctx.world_size = dist.get_world_size(group=group)
            out = x.clone()
            dist.all_reduce(out, op=dist.ReduceOp.SUM, group=group)
            return out

        @staticmethod
        def backward(ctx, grad_out):
            grad = grad_out.clone()
            dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=ctx.group)
            grad.div_(float(ctx.world_size))
            return grad, None

    return _AllReduceSumMeanGrad.apply(x, group)


def _unpack_sharded_logits(tensor: DTensor):
    """Extract local shard and global index range from a vocab-sharded DTensor.

    Returns: (group, local_logits, local_vocab_size, global_vocab_size, vocab_start, vocab_end)
    """
    mesh = tensor.device_mesh
    group = mesh.get_group()
    rank = dist.get_rank(group)
    world = dist.get_world_size(group)
    local_logits = tensor.to_local()
    local_vocab_size = local_logits.size(-1)

    # All-gather vocab sizes (shards may differ when vocab_size % world != 0)
    size_tensor = torch.tensor(local_vocab_size, device=local_logits.device, dtype=torch.int64)
    size_list = [torch.zeros_like(size_tensor) for _ in range(world)]
    dist.all_gather(size_list, size_tensor, group=group)
    sizes = [int(s.item()) for s in size_list]

    vocab_start = sum(sizes[:rank])
    vocab_end = vocab_start + sizes[rank]
    global_vocab_size = sum(sizes)
    return group, local_logits, local_vocab_size, global_vocab_size, vocab_start, vocab_end


def _sharded_logsumexp(local_logits: torch.Tensor, group, dim: int = -1, *, mean_grad: bool = False) -> torch.Tensor:
    """Numerically stable logsumexp over sharded vocab: max-shift trick.

    logsumexp(x) = max(x) + log(sum(exp(x - max(x))))
    """
    # Global max for numerical stability
    local_max = local_logits.max(dim=dim).values
    global_max = local_max.detach().clone()
    dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=group)

    # Each rank: exp(x - max) → sum → all-reduce
    local_exp_sum = torch.exp(local_logits - global_max.unsqueeze(dim)).sum(dim=dim)
    global_exp_sum = _allreduce_sum(local_exp_sum, group, mean_grad=mean_grad)

    return global_max + torch.log(global_exp_sum)


# =============================================================================
# Sharded loss/metric functions
# =============================================================================


def compute_token_log_probs_sharded(
    logits: DTensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Sharded log_prob: log_prob(label) = logit[label] - logsumexp(logits).

    Each rank checks if the label falls in its vocab shard, gathers the logit
    if so (zero otherwise), then all-reduces to get the global label logit.
    """
    group, local_logits, local_vocab_size, _, vocab_start, vocab_end = _unpack_sharded_logits(logits)
    local_logits_f32 = local_logits.float()
    if temperature != 1.0:
        local_logits_f32 = local_logits_f32 / temperature

    logsumexp = _sharded_logsumexp(local_logits_f32, group, dim=-1, mean_grad=True)

    labels = labels.to(local_logits_f32.device)
    local_idx = labels - vocab_start
    in_shard = (labels >= vocab_start) & (labels < vocab_end) & (labels != ignore_index)

    # Safe gather: clamp index to valid range, zero out non-owners after
    safe_idx = local_idx.clamp(0, max(local_vocab_size - 1, 0))
    gathered = local_logits_f32.gather(dim=-1, index=safe_idx.unsqueeze(-1)).squeeze(-1)
    local_label_logits = torch.where(in_shard, gathered, torch.zeros_like(gathered))

    # Exactly one rank owns each label; SUM gives the correct global logit
    global_label_logits = _allreduce_sum(local_label_logits, group, mean_grad=True)

    log_probs = global_label_logits - logsumexp
    return torch.where(labels == ignore_index, torch.zeros_like(log_probs), log_probs)


def compute_entropy_sharded(logits: DTensor) -> torch.Tensor:
    """Sharded entropy: H = -sum(p * log(p)) where p = softmax(logits).

    Uses a custom autograd Function for correct gradients:
    dH/dx_j = -p_j * (H + log(p_j))
    """

    class _ShardedEntropy(torch.autograd.Function):
        @staticmethod
        def forward(ctx, local_logits, group):
            dim = -1

            # Max-shift for numerical stability
            local_max = local_logits.max(dim=dim).values
            global_max = local_max.detach().clone()
            dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=group)

            shifted = local_logits - global_max.unsqueeze(dim)
            local_exp = torch.exp(shifted)

            # Global partition function Z
            Z = local_exp.sum(dim=dim).detach().clone()
            dist.all_reduce(Z, op=dist.ReduceOp.SUM, group=group)

            # p = exp(shifted) / Z, log(p) = shifted - log(Z)
            local_probs = local_exp / Z.unsqueeze(dim)
            local_log_probs = shifted - torch.log(Z).unsqueeze(dim)

            # H = -sum(p * log(p)), computed locally then all-reduced
            plogp_sum = (local_probs * local_log_probs).sum(dim=dim).detach().clone()
            dist.all_reduce(plogp_sum, op=dist.ReduceOp.SUM, group=group)
            entropy = -plogp_sum

            ctx.save_for_backward(local_probs, local_log_probs, entropy)
            return entropy

        @staticmethod
        def backward(ctx, grad_out):
            local_probs, local_log_probs, entropy = ctx.saved_tensors
            # dH/dx_j = -p_j * (H + log(p_j))
            grad = grad_out.unsqueeze(-1) * (-local_probs * (entropy.unsqueeze(-1) + local_log_probs))
            return grad, None

    group, local_logits, _, _, _, _ = _unpack_sharded_logits(logits)
    return _ShardedEntropy.apply(local_logits.float(), group)


def gather_token_logits_sharded(logits: DTensor, token_ids: torch.Tensor) -> torch.Tensor:
    """Sharded gather: extract logits for specific token_ids from vocab-sharded DTensor.

    Same shard-check + all-reduce pattern as compute_token_log_probs_sharded.
    """
    group, local_logits, _, _, vocab_start, vocab_end = _unpack_sharded_logits(logits)
    token_ids = token_ids.to(local_logits.device)

    local_idx = token_ids - vocab_start
    in_shard = (token_ids >= vocab_start) & (token_ids < vocab_end)
    safe_idx = local_idx.clamp(0, max=local_logits.size(-1) - 1)

    gathered = local_logits.index_select(dim=-1, index=safe_idx)
    masked = torch.where(in_shard, gathered, torch.zeros_like(gathered))
    return _allreduce_sum(masked, group, mean_grad=True)


def compute_kd_loss_sharded(
    logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    label: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Sharded KD loss: -sum(teacher_prob * student_log_prob), averaged over valid tokens.

    Teacher and student may independently be DTensor or plain Tensor.
    """

    def _get_local(t, vocab_start, vocab_end):
        return t.to_local() if isinstance(t, DTensor) else t[..., vocab_start:vocab_end]

    # Derive sharding metadata from whichever tensor is a DTensor
    ref = logits if isinstance(logits, DTensor) else teacher_logits
    if not isinstance(ref, DTensor):
        raise TypeError("compute_kd_loss_sharded requires at least one DTensor input")

    group, _, _, _, vocab_start, vocab_end = _unpack_sharded_logits(ref)

    local_student = _get_local(logits, vocab_start, vocab_end).float()
    local_teacher = _get_local(teacher_logits, vocab_start, vocab_end).float()

    # logsumexp: sharded path for DTensor, local for plain Tensor
    if isinstance(teacher_logits, DTensor):
        teacher_lse = _sharded_logsumexp(local_teacher, group, dim=-1, mean_grad=False)
    else:
        teacher_lse = torch.logsumexp(teacher_logits.float(), dim=-1)

    if isinstance(logits, DTensor):
        student_lse = _sharded_logsumexp(local_student, group, dim=-1, mean_grad=False)
    else:
        student_lse = torch.logsumexp(logits.float(), dim=-1)

    # Cross-entropy: sum(teacher_prob * student_log_prob) per position
    teacher_probs = torch.exp(local_teacher - teacher_lse.unsqueeze(-1))
    student_log_probs = local_student - student_lse.unsqueeze(-1)
    local_ce = (teacher_probs * student_log_probs).sum(dim=-1)
    global_ce = _allreduce_sum(local_ce, group, mean_grad=True)

    mask = (label != ignore_index).float()
    return -(global_ce.view(-1) * mask.view(-1)).sum() / mask.sum()


def compute_argmax_sharded(logits: DTensor, mask: torch.Tensor) -> torch.Tensor:
    """Sharded argmax: two-step all-reduce for global max value + index.

    Step 1: all-reduce MAX to find global max value.
    Step 2: winning ranks contribute their global index, all-reduce MIN picks first.
    """
    group, local_logits, _, global_vocab_size, vocab_start, _ = _unpack_sharded_logits(logits)
    masked_local = local_logits.detach()[mask].float()
    local_max, local_argmax = masked_local.max(dim=-1)

    # Global max value
    global_max = local_max.clone()
    dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=group)

    # Winning ranks contribute global index; others contribute sentinel (>any valid index)
    candidate = torch.where(
        local_max == global_max,
        local_argmax + vocab_start,
        torch.full_like(local_argmax, global_vocab_size),
    )
    global_argmax = candidate.clone()
    dist.all_reduce(global_argmax, op=dist.ReduceOp.MIN, group=group)
    return global_argmax
