from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import masked_mean


def masked_sum(values: torch.Tensor, mask: torch.Tensor, dim: int | tuple[int, ...] | None = None) -> torch.Tensor:
    """verl-compatible masked sum.

    NaNs outside the mask are zeroed before summation so padding-only garbage
    cannot contaminate the result.
    """
    valid_values = torch.where(mask.bool(), values, 0.0)
    return (valid_values * mask).sum(dim=dim)


def agg_loss(
    loss_mat: torch.Tensor,
    loss_mask: torch.Tensor,
    loss_agg_mode: str,
    dp_size: int = 1,
    batch_num_tokens: Optional[torch.Tensor | int] = None,
    global_batch_size: Optional[torch.Tensor | int] = None,
    loss_scale_factor: Optional[int] = None,
) -> torch.Tensor:
    """Aggregate token/sequence losses following verl's ``agg_loss`` contract.

    The returned scalar is invariant to DP/FSDP averaging when callers provide
    global batch metadata. For ``token-mean`` this is exactly:
    ``masked_sum(loss_mat, loss_mask) / batch_num_tokens * dp_size``.
    """
    if loss_agg_mode == "token-mean":
        if batch_num_tokens is None:
            if dp_size > 1:
                raise ValueError("(global) batch_num_tokens is required when dp_size > 1")
            batch_num_tokens = loss_mask.sum()
        return masked_sum(loss_mat, loss_mask) / batch_num_tokens * dp_size

    if loss_agg_mode in {"seq-mean-token-sum", "seq-mean-token-sum-norm"}:
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        seq_mask = (torch.sum(loss_mask, dim=-1) > 0).float()
        if global_batch_size is None:
            if dp_size > 1:
                raise ValueError("global_batch_size is required when dp_size > 1")
            global_batch_size = seq_mask.sum()
        loss = masked_sum(seq_losses, seq_mask) / global_batch_size * dp_size
        if loss_agg_mode == "seq-mean-token-sum-norm":
            loss /= loss_scale_factor if loss_scale_factor is not None else loss_mask.shape[-1]
        return loss

    if loss_agg_mode == "seq-mean-token-mean":
        seq_token_counts = torch.sum(loss_mask, dim=-1)
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / (seq_token_counts + 1e-8)
        seq_mask = (seq_token_counts > 0).float()
        if global_batch_size is None:
            if dp_size > 1:
                raise ValueError("global_batch_size is required when dp_size > 1")
            global_batch_size = seq_mask.sum()
        return masked_sum(seq_losses, seq_mask) / global_batch_size * dp_size

    raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")


class GPTLMLoss(nn.Module):
    """GPT Language Model Loss."""

    def __init__(self):
        super().__init__()
        self.IGNORE_INDEX = -100
        self.loss = nn.CrossEntropyLoss(ignore_index=self.IGNORE_INDEX)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        return self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


class SFTLoss(nn.Module):
    """
    SFT Loss
    """

    def __init__(self, token_level_loss: bool = True, loss_agg_mode: str = "token-mean"):
        super().__init__()
        self.token_level_loss = token_level_loss
        self.loss_agg_mode = loss_agg_mode

    def forward(
        self,
        per_token_logps: torch.Tensor,
        loss_mask: torch.Tensor,
        dp_size: int = 1,
        batch_num_tokens: Optional[torch.Tensor] = None,
        global_num_tokens: Optional[torch.Tensor] = None,
        global_batch_size: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.token_level_loss:
            if batch_num_tokens is None:
                batch_num_tokens = global_num_tokens
            return agg_loss(
                -per_token_logps,
                loss_mask,
                self.loss_agg_mode,
                dp_size=dp_size,
                batch_num_tokens=batch_num_tokens,
                global_batch_size=global_batch_size,
            )
        return masked_mean(-per_token_logps, loss_mask, dim=-1).mean()


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
        enable_vllm_is_correction: bool = False,
        vllm_is_truncated_threshold: list = None,
        vllm_is_correction_type: str = "tis",
        loss_agg_mode: str = "token-mean",
    ) -> None:
        super().__init__()
        self.clip_eps_low = clip_eps_low
        self.clip_eps_high = clip_eps_high
        self.token_level_loss = token_level_loss
        self.dual_clip = dual_clip
        self.policy_loss_type = policy_loss_type
        self.enable_vllm_is_correction = enable_vllm_is_correction
        self.vllm_is_truncated_threshold = vllm_is_truncated_threshold
        self.vllm_is_correction_type = vllm_is_correction_type
        self.loss_agg_mode = loss_agg_mode

        # GSPO requires sequence-level loss
        if policy_loss_type == "gspo":
            self.token_level_loss = False

        # Dual-clip PPO: https://arxiv.org/pdf/1912.09729
        if dual_clip is not None:
            assert dual_clip > 1.0, f"dual_clip must be > 1.0, got {dual_clip}"

        if self.vllm_is_correction_type not in {"tis", "icepop", "seq-mask-tis"}:
            raise ValueError(
                f"Invalid vllm_is_correction_type: {self.vllm_is_correction_type}, must be one of tis/icepop/seq-mask-tis"
            )

    def forward(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        rollout_log_probs: Optional[torch.Tensor] = None,
        dp_size: int = 1,
        batch_num_tokens: Optional[torch.Tensor] = None,
        global_num_tokens: Optional[torch.Tensor] = None,
        global_batch_size: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.policy_loss_type == "ppo":
            log_ratio = torch.clamp(log_probs - old_log_probs, min=-20.0, max=20.0)
            ratio = log_ratio.exp()
        elif self.policy_loss_type == "gspo":
            # GSPO: https://arxiv.org/pdf/2507.18071
            if self.enable_vllm_is_correction:
                log_ratio = log_probs - rollout_log_probs
            else:
                log_ratio = log_probs - old_log_probs
            ratio = (log_ratio * action_mask).sum(dim=-1) / action_mask.sum(dim=-1)
            ratio = ratio.exp().unsqueeze(-1) * action_mask
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
        vllm_kl = None
        if self.enable_vllm_is_correction and self.policy_loss_type == "ppo":
            low_threshold, high_threshold = self.vllm_is_truncated_threshold
            log_ratio = old_log_probs - rollout_log_probs
            if self.vllm_is_correction_type == "icepop":
                # ICEPOP: token-level filtering (set coefficients outside the interval to 0)
                vllm_is = torch.exp(log_ratio).detach()
                mask = (vllm_is >= low_threshold) & (vllm_is <= high_threshold)
                vllm_is = vllm_is * mask
                loss = vllm_is * loss
            elif self.vllm_is_correction_type == "seq-mask-tis":
                # seq-mask-tis: use sequence-level geometric mean only for filtering,
                # correction coefficients still use TIS (token-level clamp)
                seq_log_ratio = masked_mean(log_ratio, action_mask, dim=-1)
                seq_is = torch.exp(seq_log_ratio)
                seq_mask = (seq_is >= low_threshold) & (seq_is <= high_threshold)
                vllm_is = torch.exp(log_ratio).detach()
                loss = seq_mask.unsqueeze(-1) * vllm_is * loss
            else:
                # TIS: token-level clamp with low and high thresholds
                vllm_is = torch.exp(log_ratio).clamp(min=low_threshold, max=high_threshold).detach()
                loss = vllm_is * loss
            vllm_kl = masked_mean(rollout_log_probs - old_log_probs, action_mask, dim=None)

        if self.token_level_loss:
            if batch_num_tokens is None:
                batch_num_tokens = global_num_tokens
            loss = agg_loss(
                loss,
                action_mask,
                self.loss_agg_mode,
                dp_size=dp_size,
                batch_num_tokens=batch_num_tokens,
                global_batch_size=global_batch_size,
            )
        else:
            loss = agg_loss(
                loss,
                action_mask,
                "seq-mean-token-mean",
                dp_size=dp_size,
                global_batch_size=global_batch_size,
            )
        clip_ratio = masked_mean(torch.lt(surr2, surr1).float(), action_mask, dim=None)
        ppo_kl = masked_mean(-log_ratio.detach(), action_mask, dim=None)
        return loss, clip_ratio, ppo_kl, vllm_kl


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
        batch_num_tokens: Optional[torch.Tensor] = None,
        global_num_tokens: Optional[torch.Tensor] = None,
        global_batch_size: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.clip_eps is not None:
            values_clipped = old_values + (values - old_values).clamp(-self.clip_eps, self.clip_eps)
            surr1 = (values_clipped - returns) ** 2
            surr2 = (values - returns) ** 2
            loss = torch.max(surr1, surr2)
        else:
            loss = (values - returns) ** 2

        if self.token_level_loss:
            if batch_num_tokens is None:
                batch_num_tokens = global_num_tokens
            loss = agg_loss(
                loss,
                action_mask,
                "token-mean",
                dp_size=dp_size,
                batch_num_tokens=batch_num_tokens,
                global_batch_size=global_batch_size,
            )
        else:
            loss = agg_loss(
                loss,
                action_mask,
                "seq-mean-token-mean",
                dp_size=dp_size,
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
