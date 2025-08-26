from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F


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


def _logsumexp_by_chunk(logits: torch.Tensor, chunk_size: int = 1024) -> torch.Tensor:
    seq_len = logits.shape[0]
    logsumexp_values = torch.zeros((seq_len), device=logits.device, dtype=logits.dtype)
    for s_idx in range(0, seq_len, chunk_size):
        end_idx = min(s_idx + chunk_size, seq_len)
        logsumexp_values[s_idx:end_idx] = torch.logsumexp(logits[s_idx:end_idx], dim=-1)

    return logsumexp_values


def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if temperature != 1.0:
        logits.div_(temperature)
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
            row_log_probs = F.log_softmax(row_logits, dim=-1)
            row_log_probs_labels = row_log_probs.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            log_probs_labels.append(row_log_probs_labels)
        log_probs_labels = torch.stack(log_probs_labels)
    return log_probs_labels


def masked_mean(tensor: torch.Tensor, mask: Optional[torch.Tensor], dim: int = None) -> torch.Tensor:
    if mask is None:
        return tensor.mean(dim=dim)
    return (tensor * mask).sum(dim=dim) / mask.sum(dim=dim)


def masked_normalize(tensor: torch.Tensor, mask: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    tensor = tensor * mask
    mean = masked_mean(tensor, mask, dim=dim)
    mean_centered = tensor - mean
    var = masked_mean(mean_centered**2, mask, dim=dim)
    return mean_centered * var.clamp(min=eps).rsqrt()


@torch.compile
def compute_entropy(logits: torch.Tensor):
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    return entropy
