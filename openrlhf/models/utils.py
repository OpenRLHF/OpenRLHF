from typing import Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor


# ============================================================
# Distributed Primitives (Private Helpers)
# ============================================================


def _all_reduce_sum(x: torch.Tensor, group, *, mean_grad: bool = False) -> torch.Tensor:
    """All-reduce SUM with selectable backward semantics.

    Args:
        x: Tensor to all-reduce.
        group: Process group (typically a TP group).
        mean_grad: If True, backward returns the *mean* of grad contributions across
            ranks (useful when the downstream loss is replicated on every TP rank).
            If False, backward returns the *sum* across ranks (useful when each rank
            holds a distinct partial contribution to the upstream gradient).
    """
    if not dist.is_initialized():
        return x

    if not mean_grad:
        return dist_nn.all_reduce(x, op=dist.ReduceOp.SUM, group=group)

    # mean_grad=True: all-reduce SUM in forward, but average gradients in backward.
    # This avoids the extra tp_size scaling when the loss is replicated across TP ranks.
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


def _get_dtensor_shard_info(tensor: DTensor):
    """Extract sharding info from a vocab-sharded DTensor."""
    mesh = tensor.device_mesh
    group = mesh.get_group()
    rank = dist.get_rank(group)
    world = dist.get_world_size(group)
    local = tensor.to_local()
    local_vocab = local.size(-1)

    size_tensor = torch.tensor(local_vocab, device=local.device, dtype=torch.int64)
    size_list = [torch.zeros_like(size_tensor) for _ in range(world)]
    dist.all_gather(size_list, size_tensor, group=group)
    sizes = [int(s.item()) for s in size_list]
    vocab_start = sum(sizes[:rank])
    vocab_end = vocab_start + sizes[rank]
    global_vocab = sum(sizes)
    return group, local, local_vocab, global_vocab, vocab_start, vocab_end


def _logsumexp_sharded(
    logits_local: torch.Tensor,
    group,
    dim: int = -1,
    *,
    mean_grad: bool = False,
) -> torch.Tensor:
    """Compute logsumexp over sharded vocab dimension."""
    local_max = logits_local.max(dim=dim).values
    # `global_max` is used only for numerical stability: logsumexp(x) is shift-invariant,
    # so we can safely stop gradients through this MAX reduction.
    global_max = local_max.detach().clone()
    dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=group)
    local_exp = torch.exp(logits_local - global_max.unsqueeze(dim))
    local_exp_sum = local_exp.sum(dim=dim)
    # NOTE: Different call sites need different backward semantics:
    # - For log-prob / label-logit use cases, upstream grads are typically identical across TP ranks,
    #   so SUM-backward would introduce an extra tp_size scaling. Use mean_grad=True there.
    # - For softmax-probability-derived quantities (e.g., entropy/KD), upstream grads contain
    #   rank-local partial contributions that must be summed. Use mean_grad=False there.
    global_exp_sum = _all_reduce_sum(local_exp_sum, group, mean_grad=mean_grad)
    return global_max + torch.log(global_exp_sum)


def _local_slice_by_vocab(logits: torch.Tensor, vocab_start: int, vocab_end: int) -> torch.Tensor:
    """Get local vocab slice from logits (handles both DTensor and regular Tensor)."""
    if isinstance(logits, DTensor):
        return logits.to_local()
    return logits[..., vocab_start:vocab_end]


def _logsumexp_by_chunk(logits: torch.Tensor, chunk_size: int = 1024) -> torch.Tensor:
    """Compute logsumexp in chunks to reduce peak memory."""
    seq_len = logits.shape[0]
    logsumexp_values = torch.zeros((seq_len), device=logits.device, dtype=logits.dtype)
    for s_idx in range(0, seq_len, chunk_size):
        end_idx = min(s_idx + chunk_size, seq_len)
        logsumexp_values[s_idx:end_idx] = torch.logsumexp(logits[s_idx:end_idx], dim=-1)

    return logsumexp_values


# ============================================================
# Log Probs
# ============================================================


def log_probs_from_sharded_logits(
    logits: DTensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Compute log probs from vocab-sharded DTensor logits."""
    group, local_logits, local_vocab, _, vocab_start, vocab_end = _get_dtensor_shard_info(logits)
    logits_local = local_logits.float()
    if temperature != 1.0:
        logits_local = logits_local / temperature

    logsumexp_global = _logsumexp_sharded(logits_local, group, dim=-1, mean_grad=True)

    labels = labels.to(logits_local.device)
    local_labels = labels - vocab_start
    in_shard = (labels >= vocab_start) & (labels < vocab_end) & (labels != ignore_index)
    safe_local = local_labels.clamp(0, max(local_vocab - 1, 0))
    gathered = logits_local.gather(dim=-1, index=safe_local.unsqueeze(-1)).squeeze(-1)
    # Use SUM reduction with masking so gradients route only to the owning vocab shard.
    local_label_logits = torch.where(in_shard, gathered, torch.zeros_like(gathered))

    global_label_logits = _all_reduce_sum(local_label_logits, group, mean_grad=True)

    log_probs = global_label_logits - logsumexp_global
    log_probs = torch.where(labels == ignore_index, torch.zeros_like(log_probs), log_probs)
    return log_probs


def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Compute log probs from logits (supports both DTensor and regular Tensor)."""
    if isinstance(logits, DTensor):
        return log_probs_from_sharded_logits(logits, labels, temperature)
    if temperature != 1.0:
        logits = logits / temperature  # Avoid in-place op to prevent modifying caller's tensor
    # https://github.com/OpenRLHF/OpenRLHF/pull/718#issuecomment-2641081881
    if logits.dtype in [torch.float32, torch.float64]:
        batch_dim = logits.shape[:-1]
        last_dim = logits.shape[-1]
        try:
            from flash_attn.ops.triton.cross_entropy import cross_entropy_loss

            output = cross_entropy_loss(logits.reshape(-1, last_dim), labels.reshape(-1))
            log_probs_labels = -output[0].view(*batch_dim)
        except ImportError:
            logits_labels = torch.gather(logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
            logsumexp_values = _logsumexp_by_chunk(logits.reshape(-1, last_dim))
            logsumexp_values = logsumexp_values.view(*batch_dim)
            log_probs_labels = logits_labels - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        log_probs_labels = []
        for row_logits, row_labels in zip(logits, labels):  # loop to reduce peak mem consumption
            row_log_probs = F.log_softmax(row_logits.float(), dim=-1)
            row_log_probs_labels = row_log_probs.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            log_probs_labels.append(row_log_probs_labels)
        log_probs_labels = torch.stack(log_probs_labels)
    return log_probs_labels


# ============================================================
# Token Logits Selection
# ============================================================


def select_token_logits(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """Select logits for specific token ids, supports DTensor-sharded vocab."""
    if not isinstance(logits, DTensor):
        token_ids = token_ids.to(device=logits.device)
        return logits.index_select(dim=-1, index=token_ids)

    group, local_logits, _, _, vocab_start, vocab_end = _get_dtensor_shard_info(logits)
    token_ids = token_ids.to(local_logits.device)
    local_indices = token_ids - vocab_start
    in_shard = (token_ids >= vocab_start) & (token_ids < vocab_end)
    safe_local = local_indices.clamp(0, max=local_logits.size(-1) - 1)
    gathered = local_logits.index_select(dim=-1, index=safe_local)
    # Use SUM reduction so gradients propagate only to ranks that own the selected tokens.
    local_values = torch.where(in_shard, gathered, torch.zeros_like(gathered))

    return _all_reduce_sum(local_values, group, mean_grad=True)


# ============================================================
# Entropy
# ============================================================


def _compute_entropy_sharded(logits: DTensor) -> torch.Tensor:
    """Entropy for vocab-sharded DTensor logits with custom backward.

    We avoid relying on generic collective autograd for the softmax denominator,
    because the "right" backward behavior depends on whether the upstream gradient
    is a sum of rank-local contributions vs an identical replicated tensor.

    This implementation computes entropy via:
      H = -sum_i p_i log p_i
    and uses a closed-form gradient:
      dH/dx_k = -p_k * (H + log p_k)
    where p_k is the global softmax probability.
    """

    class _EntropyFromVocabShardedLogits(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x_local: torch.Tensor, group) -> torch.Tensor:  # type: ignore[override]
            # x_local: (..., local_vocab)
            dim = -1
            local_max = x_local.max(dim=dim).values
            global_max = local_max.detach().clone()
            dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=group)

            shifted = x_local - global_max.unsqueeze(dim)
            local_exp = torch.exp(shifted)
            local_exp_sum = local_exp.sum(dim=dim)

            global_exp_sum = local_exp_sum.detach().clone()
            dist.all_reduce(global_exp_sum, op=dist.ReduceOp.SUM, group=group)

            # p = exp(x - m) / sum(exp(x - m))
            probs_local = local_exp / global_exp_sum.unsqueeze(dim)
            log_p_local = shifted - torch.log(global_exp_sum).unsqueeze(dim)

            # H = -sum_i p_i log p_i
            local_plogp = (probs_local * log_p_local).sum(dim=dim)
            global_plogp = local_plogp.detach().clone()
            dist.all_reduce(global_plogp, op=dist.ReduceOp.SUM, group=group)
            entropy = -global_plogp

            ctx.save_for_backward(probs_local, log_p_local, entropy)
            return entropy

        @staticmethod
        def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
            probs_local, log_p_local, entropy = ctx.saved_tensors
            # dH/dx_k = -p_k * (H + log p_k)
            grad_x = grad_out.unsqueeze(-1) * (-probs_local * (entropy.unsqueeze(-1) + log_p_local))
            return grad_x, None

    group, local_logits, _, _, _, _ = _get_dtensor_shard_info(logits)
    return _EntropyFromVocabShardedLogits.apply(local_logits.float(), group)


@torch.compile
def _compute_entropy_dense(logits: torch.Tensor):
    """Compute entropy for dense (non-sharded) logits."""
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    return entropy


def compute_entropy(logits: torch.Tensor):
    """Compute entropy (supports both DTensor and regular Tensor)."""
    if isinstance(logits, DTensor):
        return _compute_entropy_sharded(logits)
    return _compute_entropy_dense(logits)


# ============================================================
# KL Divergence & Reward
# ============================================================


def compute_approx_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    kl_estimator: str = "k1",
) -> torch.Tensor:
    """
    Compute the approximate KL divergence between two distributions.
    Schulman blog: http://joschu.net/blog/kl-approx.html

    Args:
        log_probs: Log probabilities of the new distribution.
        log_probs_base: Log probabilities of the base distribution.
    """

    if kl_estimator == "k1":
        log_ratio = log_probs.float() - log_probs_base.float()

    # The k2 estimator is the non negative kl approximation in
    # http://joschu.net/blog/kl-approx.html
    # The k2_loss is approximately equivalent to the
    # one-step KL divergence penalty with the k1 estimator
    # used in https://arxiv.org/pdf/2310.10505.
    if kl_estimator == "k2":
        log_ratio = log_probs.float() - log_probs_base.float()
        log_ratio = log_ratio**2 / 2.0

    # The k3 estimator is the non negative kl approximation in
    # http://joschu.net/blog/kl-approx.html
    if kl_estimator == "k3":
        log_ratio = log_probs.float() - log_probs_base.float()
        log_ratio = -log_ratio
        log_ratio = log_ratio.exp() - 1 - log_ratio

    log_ratio = log_ratio.clamp(min=-10, max=10)
    return log_ratio


def compute_reward(
    r: Union[torch.Tensor, float],
    kl_coef: float,
    kl: Union[torch.Tensor, list[torch.Tensor]],
    action_mask: Optional[torch.Tensor] = None,
    reward_clip_range: Tuple[float, float] = None,
) -> Union[torch.Tensor, list[torch.Tensor]]:
    """Compute reward with KL penalty."""
    if kl_coef <= 0.0:
        kl_coef = 0.0

    if reward_clip_range:
        r = r.clamp(min=reward_clip_range[0], max=reward_clip_range[1])

    kl_reward = -kl_coef * kl
    # The following code is equivalent to:
    #
    # last_reward = torch.zeros_like(kl)
    # for i in range(last_reward.size(0)):
    #     for t in reversed(range(last_reward.size(1))):
    #         if action_mask[i][t] > 0.5:
    #             last_reward[i][t] = r[i]
    #             break
    #
    eos_indices = action_mask.size(1) - 1 - action_mask.long().fliplr().argmax(dim=1, keepdim=True)
    last_reward = torch.zeros_like(kl).scatter_(dim=1, index=eos_indices, src=r.unsqueeze(1).to(kl.dtype))

    reward = last_reward + kl_reward

    return reward


# ============================================================
# Knowledge Distillation
# ============================================================


def kd_loss_from_logits(
    logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    label: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """KD loss that supports DTensor-sharded vocab logits."""
    if not isinstance(logits, DTensor) and not isinstance(teacher_logits, DTensor):
        teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
        inf_mask = torch.isinf(logits)
        logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
        prod_probs = torch.masked_fill(teacher_probs * logprobs, inf_mask, 0)
        x = torch.sum(prod_probs, dim=-1).view(-1)
        mask = (label != ignore_index).int()
        return -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)

    ref = logits if isinstance(logits, DTensor) else teacher_logits
    group, _, _, _, vocab_start, vocab_end = _get_dtensor_shard_info(ref)

    student_local = _local_slice_by_vocab(logits, vocab_start, vocab_end).float()
    teacher_local = _local_slice_by_vocab(teacher_logits, vocab_start, vocab_end).float()

    if isinstance(teacher_logits, DTensor):
        teacher_logsumexp = _logsumexp_sharded(teacher_local, group, dim=-1, mean_grad=False)
    else:
        teacher_logsumexp = torch.logsumexp(teacher_logits.float(), dim=-1)

    if isinstance(logits, DTensor):
        student_logsumexp = _logsumexp_sharded(student_local, group, dim=-1, mean_grad=False)
    else:
        student_logsumexp = torch.logsumexp(logits.float(), dim=-1)

    teacher_probs_local = torch.exp(teacher_local - teacher_logsumexp.unsqueeze(-1))
    student_logprobs_local = student_local - student_logsumexp.unsqueeze(-1)
    local_sum = (teacher_probs_local * student_logprobs_local).sum(dim=-1)

    total_sum = _all_reduce_sum(local_sum, group, mean_grad=True)

    mask = (label != ignore_index).float()
    return -(total_sum.view(-1) * mask.view(-1)).sum() / mask.sum()


# ============================================================
# Masking Utilities
# ============================================================


def masked_mean(tensor: torch.Tensor, mask: Optional[torch.Tensor], dim: int = None) -> torch.Tensor:
    """Compute mean with optional mask."""
    if mask is None:
        return tensor.mean(dim=dim)
    return (tensor * mask).sum(dim=dim) / mask.sum(dim=dim)


def masked_normalize(tensor: torch.Tensor, mask: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    """Normalize tensor with mask (zero mean, unit variance)."""
    tensor = tensor * mask
    mean = masked_mean(tensor, mask, dim=dim)
    mean_centered = tensor - mean
    var = masked_mean(mean_centered**2, mask, dim=dim)
    return mean_centered * var.clamp(min=eps).rsqrt()
