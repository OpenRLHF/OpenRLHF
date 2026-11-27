from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .utils import masked_mean


def aggregate_loss(
    loss: torch.Tensor,
    loss_mask: torch.Tensor,
    token_level_loss: bool = True,
    dp_size: int = 1,
    batch_num_tokens: Optional[float] = None,
    global_batch_size: Optional[float] = None,
) -> torch.Tensor:
    """Aggregate a per-token loss matrix into a scalar using one of two reduction modes:

    - ``token_level_loss=True``  -> per-token: masked-sum / global token count.
    - ``token_level_loss=False`` -> per-sample: sum of per-sequence token-means / global
      sample count.

    ``batch_num_tokens`` (token mode) and ``global_batch_size`` (sample mode) carry the
    *global* batch totals so the loss is invariant to data-parallel sharding; ``dp_size``
    compensates for the gradient averaging that DeepSpeed/DDP applies across DP ranks.
    """
    if token_level_loss:
        if batch_num_tokens is None:
            return masked_mean(loss, loss_mask, dim=None)
        return (loss * loss_mask).sum() / batch_num_tokens * dp_size

    token_counts = loss_mask.sum(dim=-1)
    seq_loss = (loss * loss_mask).sum(dim=-1) / (token_counts + 1e-8)
    seq_mask = (token_counts > 0).float()  # exclude fully masked sequences
    if global_batch_size is None:
        return masked_mean(seq_loss, seq_mask, dim=None)
    return (seq_loss * seq_mask).sum() / global_batch_size * dp_size


class GPTLMLoss(nn.Module):
    """
    GPT Language Model Loss
    """

    def __init__(self, ring_attn_group=None):
        super().__init__()
        self.IGNORE_INDEX = -100
        self.loss = nn.CrossEntropyLoss(ignore_index=self.IGNORE_INDEX)

        self.ring_attn_group = ring_attn_group
        if self.ring_attn_group:
            self.ring_attn_rank = dist.get_rank(self.ring_attn_group)
            self.ring_attn_world_size = dist.get_world_size(self.ring_attn_group)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # RingAttention
        if self.ring_attn_group is not None:
            total_seq_len = labels.size(-1)
            seq_len_per_process = total_seq_len // self.ring_attn_world_size
            start_idx = self.ring_attn_rank * seq_len_per_process
            end_idx = min(start_idx + seq_len_per_process, total_seq_len)
            labels = labels[..., start_idx:end_idx]

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # if labels are all IGNORE_INDEX, then nn.CrossEntropyLoss will be nan
            if torch.all(shift_labels == self.IGNORE_INDEX):
                # Use mean of logits multiplied by 0 to maintain gradient flow
                loss = shift_logits.mean() * 0
            else:
                loss = self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            dist.all_reduce(loss, op=dist.ReduceOp.SUM, group=self.ring_attn_group)
            loss = loss / self.ring_attn_world_size
        else:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return loss


class SFTLoss(nn.Module):
    """
    SFT Loss
    """

    def __init__(self, token_level_loss: bool = True):
        super().__init__()
        self.token_level_loss = token_level_loss

    def forward(
        self,
        per_token_logps: torch.Tensor,
        loss_mask: torch.Tensor,
        dp_size: int = 1,
        batch_num_tokens: Optional[float] = None,
        global_batch_size: Optional[float] = None,
    ) -> torch.Tensor:
        loss = aggregate_loss(
            -per_token_logps,
            loss_mask,
            token_level_loss=self.token_level_loss,
            dp_size=dp_size,
            batch_num_tokens=batch_num_tokens,
            global_batch_size=global_batch_size,
        )

        return loss


class PolicyLoss(nn.Module):
    """
    Policy Loss for PPO
    """

    def __init__(
        self,
        clip_eps_low: float = 0.2,
        clip_eps_high: float = 0.2,
        dual_clip: float = None,
        token_level_loss: bool = True,
        policy_loss_type: str = "ppo",
        is_correction_level: str = "off",
        is_correction_mode: str = "mask",
        is_correction_gating: str = "ratio",
        is_correction_threshold: list = (0.5, 5.0),
    ) -> None:
        super().__init__()
        self.clip_eps_low = clip_eps_low
        self.clip_eps_high = clip_eps_high
        self.token_level_loss = token_level_loss
        self.dual_clip = dual_clip
        self.policy_loss_type = policy_loss_type
        # Train/rollout (DeepSpeed-actor vs vLLM) logprob-mismatch correction on the per-token
        # IS ratio pi_train/pi_rollout. A gate reads a statistic per unit (level: token, or the
        # per-sequence mean) and drops (mask) or clamps (clip) units outside [low, high].
        # gating selects the statistic: the ratio itself (TIS / ICEPOP / seq-mask-tis) or the
        # sampled-token binary_kl / tv divergence between rollout and train policy (a trust
        # region; FlashREINFORCE gates the per-sequence mean binary KL).
        self.is_correction_level = is_correction_level
        self.is_correction_mode = is_correction_mode
        self.is_correction_gating = is_correction_gating
        self.is_correction_threshold = is_correction_threshold

        # GSPO requires sequence-level loss (per-sample mean)
        if policy_loss_type == "gspo":
            self.token_level_loss = False

        # Dual-clip PPO: https://arxiv.org/pdf/1912.09729
        if dual_clip is not None:
            assert dual_clip > 1.0, f"dual_clip must be > 1.0, got {dual_clip}"

        if is_correction_level not in {"off", "token", "seq"}:
            raise ValueError(f"is_correction_level must be off/token/seq, got {is_correction_level}")
        if is_correction_mode not in {"mask", "clip"}:
            raise ValueError(f"is_correction_mode must be mask/clip, got {is_correction_mode}")
        if is_correction_gating not in {"ratio", "binary_kl", "tv"}:
            raise ValueError(f"is_correction_gating must be ratio/binary_kl/tv, got {is_correction_gating}")
        # Only the per-token ratio is a weight that can be clamped; a per-sequence statistic or a
        # divergence is a rejection filter.
        if is_correction_mode == "clip" and (is_correction_level == "seq" or is_correction_gating != "ratio"):
            raise ValueError(
                "is_correction_mode=clip requires is_correction_level=token and is_correction_gating=ratio"
            )
        # A divergence is bounded from above only; the default ratio band [0.5, 5] would reject everything.
        if is_correction_gating != "ratio" and is_correction_level != "off" and is_correction_threshold[0] > 0:
            raise ValueError(f"is_correction_gating={is_correction_gating} takes an upper bound only (a single delta)")

    def forward(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        rollout_log_probs: Optional[torch.Tensor] = None,
        dp_size: int = 1,
        batch_num_tokens: Optional[float] = None,
        global_batch_size: Optional[float] = None,
    ) -> torch.Tensor:
        raw_policy_log_ratio = log_probs - old_log_probs
        if self.policy_loss_type == "ppo":
            policy_log_ratio = raw_policy_log_ratio.clamp(min=-20.0, max=20.0)
            ratio = policy_log_ratio.exp()
        elif self.policy_loss_type == "gspo":
            # GSPO: https://arxiv.org/pdf/2507.18071
            if self.is_correction_level != "off":
                log_ratio = log_probs - rollout_log_probs
            else:
                log_ratio = raw_policy_log_ratio
            seq_log_ratio = (log_ratio * action_mask).sum(dim=-1) / action_mask.sum(dim=-1).clamp(min=1)
            # Bound the exponent, as the ppo branch above does. Without this a sequence
            # with a large log ratio makes ratio inf, and torch.min below then turns that
            # into 0 * inf = nan in the backward pass. Since every sequence shares the
            # actor weights, one such sequence is enough to nan the whole update.
            ratio = seq_log_ratio.clamp(min=-20.0, max=20.0).exp().unsqueeze(-1) * action_mask
        else:
            raise ValueError(f"Invalid policy loss type: {self.policy_loss_type}")

        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.clip_eps_low, 1 + self.clip_eps_high) * advantages

        if self.dual_clip is None:
            # Standard PPO
            loss = -torch.min(surr1, surr2)
        else:
            # Standard PPO clipping
            clip1 = torch.min(surr1, surr2)
            # Dual-clip: additional lower bound for negative advantages
            clip2 = torch.max(clip1, self.dual_clip * advantages)
            # Apply dual-clip: use clip2 for negative advantages, clip1 for positive advantages
            loss = -torch.where(advantages < 0, clip2, clip1)

        # Your Efficient RL Framework Secretly Brings You Off-Policy RL Training: https://fengyao.notion.site/off-policy-rl
        vllm_kl = is_filter_ratio = None
        if self.is_correction_level != "off" and self.policy_loss_type == "ppo":
            low, high = self.is_correction_threshold
            seq_level = self.is_correction_level == "seq"
            is_log_ratio = (old_log_probs - rollout_log_probs).detach()  # log(pi_train / pi_rollout)
            token_is = is_log_ratio.exp()
            # Gated statistic per unit: at seq level the ratio's geometric mean, a divergence's plain mean.
            if self.is_correction_gating == "ratio":
                stat = masked_mean(is_log_ratio, action_mask, dim=-1).unsqueeze(-1) if seq_level else is_log_ratio
                stat = stat.exp()
            else:
                p = rollout_log_probs.exp().clamp(1e-6, 1 - 1e-6)  # clamp keeps the binary-KL logs finite
                q = old_log_probs.exp().clamp(1e-6, 1 - 1e-6)
                if self.is_correction_gating == "binary_kl":
                    stat = p * (p / q).log() + (1 - p) * ((1 - p) / (1 - q)).log()
                else:  # tv
                    stat = (p - q).abs()
                if seq_level:
                    stat = masked_mean(stat, action_mask, dim=-1).unsqueeze(-1)
            if self.is_correction_mode == "clip":
                coef = stat.clamp(min=low, max=high)
                filtered = (stat < low) | (stat > high)
            else:  # mask: drop out-of-band units, survivors keep their per-token IS weight
                keep = (stat >= low) & (stat <= high)
                coef = torch.where(keep, token_is, 0.0)
                filtered = ~keep
            loss = coef * loss
            # Filter fraction at the unit's own granularity: per token, or per sequence.
            if seq_level:
                is_filter_ratio = filtered.float().mean()
            else:
                is_filter_ratio = masked_mean(filtered.float(), action_mask, dim=None)
            vllm_kl = masked_mean(rollout_log_probs - old_log_probs, action_mask, dim=None)

        loss = aggregate_loss(
            loss,
            action_mask,
            token_level_loss=self.token_level_loss,
            dp_size=dp_size,
            batch_num_tokens=batch_num_tokens,
            global_batch_size=global_batch_size,
        )
        clip_ratio = masked_mean(torch.lt(surr2, surr1).float(), action_mask, dim=None)
        ppo_kl = masked_mean(-raw_policy_log_ratio.detach(), action_mask, dim=None)
        return loss, clip_ratio, ppo_kl, vllm_kl, is_filter_ratio


class ValueLoss(nn.Module):
    """
    Value Loss for PPO
    """

    def __init__(self, clip_eps: float = None, token_level_loss: bool = True) -> None:
        super().__init__()
        self.clip_eps = clip_eps
        self.token_level_loss = token_level_loss

    def forward(
        self,
        values: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        dp_size: int = 1,
        batch_num_tokens: Optional[float] = None,
        global_batch_size: Optional[float] = None,
    ) -> torch.Tensor:
        if self.clip_eps is not None:
            values_clipped = old_values + (values - old_values).clamp(-self.clip_eps, self.clip_eps)
            surr1 = (values_clipped - returns) ** 2
            surr2 = (values - returns) ** 2
            loss = torch.max(surr1, surr2)
        else:
            loss = (values - returns) ** 2

        loss = aggregate_loss(
            loss,
            action_mask,
            token_level_loss=self.token_level_loss,
            dp_size=dp_size,
            batch_num_tokens=batch_num_tokens,
            global_batch_size=global_batch_size,
        )
        return 0.5 * loss


class PairWiseLoss(nn.Module):
    """
    Pairwise Loss for Reward Model
    """

    def forward(
        self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor, margin: torch.Tensor = None
    ) -> torch.Tensor:
        if margin is not None:
            loss = -F.logsigmoid(chosen_reward - reject_reward - margin)
        else:
            loss = -F.logsigmoid(chosen_reward - reject_reward)
        return loss.mean()


class LogExpLoss(nn.Module):
    """
    Pairwise Loss for Reward Model
    Details: https://arxiv.org/abs/2204.05862
    """

    def forward(
        self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor, margin: torch.Tensor = None
    ) -> torch.Tensor:
        loss = torch.log(1 + torch.exp(reject_reward - chosen_reward)).mean()
        return loss


class DPOLoss(nn.Module):
    """
    DPO Loss
    """

    def __init__(self, beta: float, label_smoothing: float = 0.0, ipo: bool = False) -> None:
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.ipo = ipo

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios

        if self.ipo:
            losses = (logits - 1 / (2 * self.beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
        else:
            # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )

        loss = losses.mean()
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return loss, chosen_rewards, rejected_rewards
