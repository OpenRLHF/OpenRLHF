from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .utils import masked_mean


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

    def forward(self, per_token_logps: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
        loss = (
            masked_mean(-per_token_logps, loss_mask, dim=None)
            if self.token_level_loss
            else masked_mean(-per_token_logps, loss_mask, dim=-1).mean()
        )

        return loss


class PolicyLoss(nn.Module):
    """
    Policy Loss for PPO
    """

    def __init__(self, clip_eps: float = 0.2, token_level_loss: bool = True) -> None:
        super().__init__()
        self.clip_eps = clip_eps
        self.token_level_loss = token_level_loss

    def forward(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ratio = (log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
        loss = -torch.min(surr1, surr2)
        loss = (
            masked_mean(loss, action_mask, dim=None)
            if self.token_level_loss
            else masked_mean(loss, action_mask, dim=-1).mean()
        )
        return loss


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
    ) -> torch.Tensor:
        if self.clip_eps is not None:
            values_clipped = old_values + (values - old_values).clamp(-self.clip_eps, self.clip_eps)
            surr1 = (values_clipped - returns) ** 2
            surr2 = (values - returns) ** 2
            loss = torch.max(surr1, surr2)
        else:
            loss = (values - returns) ** 2

        loss = (
            masked_mean(loss, action_mask, dim=None)
            if self.token_level_loss
            else masked_mean(loss, action_mask, dim=-1).mean()
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


# Adapted from https://github.com/ContextualAI/HALOs/blob/ca9b7e3eeea220c0944ad8095d641da33f907a7e/trainers.py#L742
class VanillaKTOLoss(nn.Module):
    """
    KTO loss for even sampling
    """

    def __init__(self, beta: float) -> None:
        super().__init__()
        self.beta = beta

    def forward(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        chosen_KL = (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
        rejected_KL = (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)

        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        rejected_logratios = policy_rejected_logps - reference_rejected_logps

        losses = torch.cat(
            (
                1 - F.sigmoid(self.beta * (chosen_logratios - rejected_KL)),
                1 - F.sigmoid(self.beta * (chosen_KL - rejected_logratios)),
            ),
            0,
        ).mean()

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()
        return losses, chosen_rewards, rejected_rewards


# Adapted from https://github.com/ContextualAI/HALOs/blob/ca9b7e3eeea220c0944ad8095d641da33f907a7e/trainers.py#L770
class KTOLoss(nn.Module):
    """
    KTO loss for uneven sampling
    """

    def __init__(
        self, beta: float, desirable_weight: float, undesirable_weight: float, world_size: int, device: torch.device
    ) -> None:
        super().__init__()
        self.beta = beta
        self.world_size = world_size
        self.device = device
        self.desirable_weight = desirable_weight
        self.undesirable_weight = undesirable_weight

    def forward(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        policy_KL_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_KL_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        KL = (policy_KL_logps - reference_KL_logps).mean().detach()
        # all_reduce sums up the KL estimates across all devices (gradient will also be scaled by world size)
        dist.all_reduce(KL, op=dist.ReduceOp.SUM)
        # take average (will also scale gradients appropriately)
        KL = (KL / self.world_size).clamp(min=0)

        if policy_chosen_logps.shape[0] != 0:
            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            chosen_losses = 1 - F.sigmoid(self.beta * (chosen_logratios - KL))
            chosen_rewards = self.beta * chosen_logratios.detach()
        else:
            # important to cast to policy_dtype; otherwise error will occur during all_gather
            chosen_losses = torch.Tensor([]).to(policy_rejected_logps.dtype).to(self.device)
            chosen_rewards = torch.Tensor([]).to(policy_rejected_logps.dtype).to(self.device)

        if policy_rejected_logps.shape[0] != 0:
            rejected_logratios = policy_rejected_logps - reference_rejected_logps
            rejected_losses = 1 - F.sigmoid(self.beta * (KL - rejected_logratios))
            rejected_rewards = self.beta * rejected_logratios.detach()
        else:
            # important to cast to policy_dtype; otherwise error will occur during all_gather
            rejected_losses = torch.Tensor([]).to(policy_chosen_logps.dtype).to(self.device)
            rejected_rewards = torch.Tensor([]).to(policy_chosen_logps.dtype).to(self.device)

        losses = torch.cat(
            (self.desirable_weight * chosen_losses, self.undesirable_weight * rejected_losses), 0
        ).mean()
        return losses, chosen_rewards, rejected_rewards, KL


# Adapted from https://github.com/microsoft/LMOps/blob/main/minillm/finetune.py#L166
class KDLoss(nn.Module):
    """
    Language Model Knowledge Distillation Loss
    """

    def __init__(self):
        super().__init__()
        self.IGNORE_INDEX = -100

    def forward(self, logits: torch.Tensor, teacher_logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
        inf_mask = torch.isinf(logits)
        logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
        prod_probs = torch.masked_fill(teacher_probs * logprobs, inf_mask, 0)
        x = torch.sum(prod_probs, dim=-1).view(-1)
        mask = (label != self.IGNORE_INDEX).int()
        distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)

        return distil_loss


class PRMLoss(nn.Module):
    """
    Process Reward Model Loss
    """

    def __init__(self, placeholder_token_id: int, reward_token_ids: Optional[list[int]] = None):
        super().__init__()
        self.IGNORE_INDEX = -100
        self.loss = nn.CrossEntropyLoss(ignore_index=self.IGNORE_INDEX)
        self.placeholder_token_id = placeholder_token_id
        self.reward_token_ids = reward_token_ids

    def forward(self, inputs: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor, *, return_acc: bool = False):
        placeholder_mask = inputs == self.placeholder_token_id
        logits = logits[placeholder_mask].squeeze(1)
        labels = labels[placeholder_mask]

        if labels.dtype == torch.float:
            # soft label
            assert len(self.reward_token_ids) == 2, "reward_token_ids should have 2 tokens for soft labels"
            logits = logits[..., self.reward_token_ids]
            positive_labels = labels.to(logits.dtype)
            negative_labels = 1 - positive_labels
            negative_labels[positive_labels != -100] = 1 - positive_labels[positive_labels != -100]
            labels = torch.stack([positive_labels, negative_labels], dim=-1)
        elif self.reward_token_ids is not None:
            # hard label with reward_token_ids set. (otherwise the whole vocab will be trained together.)
            logits = logits[..., self.reward_token_ids]
            # this is slow....
            for i, token in enumerate(self.reward_token_ids):
                labels = torch.where(labels == token, i, labels)

        loss = self.loss(logits, labels)
        if not return_acc:
            return loss

        if labels.dtype == logits.dtype:
            labels = labels.argmax(dim=-1)
        acc = (logits.argmax(dim=-1) == labels).float().mean()
        return loss, acc


class DisCOHelper:
    @staticmethod
    def calculate_scores(
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        scoring_func: str,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calculates scores s_theta(o, q) based on the chosen scoring function.
        log_probs: Log probabilities from the current policy.
        old_log_probs: Log probabilities from the old policy (for l_ratio).
        scoring_func: 'log_l' for log-likelihood, 'l_ratio' for likelihood ratio.
        action_mask: Mask for valid actions.
        """
        if scoring_func == "log_l":
            # s_theta(o,q) = log_pi_theta(o|q)
            # Sum log_probs over the sequence length, considering the action_mask
            if action_mask is not None:
                return (log_probs * action_mask).sum(dim=-1) / action_mask.sum(dim=-1)
            else:
                return log_probs.sum(dim=-1)
        elif scoring_func == "l_ratio":
            # s_theta(o,q) = pi_theta(o|q) / pi_old(o|q)
            # Ratio of probabilities (exp of log_probs difference)
            # Sum over the sequence length is not appropriate here as per original DisCO paper,
            # it's a ratio of full sequence probabilities.
            # However, in practice, using sum of log_probs (average) is more stable.
            # For this implementation, we'll use the sum of log_probs for stability,
            # effectively (log_probs - old_log_probs).exp().mean(dim=-1) if we were to average ratios
            # or (log_probs.sum - old_log_probs.sum).exp() for product of ratios.
            # The paper implies product of ratios, so exp(sum(log_probs) - sum(old_log_probs)).
            # Let's stick to the definition s_theta(o,q) = log pi_theta(o|q) for log_l
            # and s_theta(o,q) = log (pi_theta(o|q) / pi_old(o|q)) for l_ratio for numerical stability
            # This means for l_ratio, score = sum(log_probs) - sum(old_log_probs)
            current_seq_log_probs = (
                (log_probs * action_mask).sum(dim=-1) if action_mask is not None else log_probs.sum(dim=-1)
            )
            old_seq_log_probs = (
                (old_log_probs * action_mask).sum(dim=-1) if action_mask is not None else old_log_probs.sum(dim=-1)
            )
            return current_seq_log_probs - old_seq_log_probs  # log (pi_theta / pi_old)
        else:
            raise ValueError(f"Unknown scoring_func: {scoring_func}")

    @staticmethod
    def calculate_kl_penalty(
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        beta: float,
        delta: float,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the KL divergence D_KL(pi_old || pi_theta) and the penalty term.
        KL divergence is E_pi_old[log(pi_old / pi_theta)].
        Penalty is beta * [KL - delta]_+^2.

        log_probs: Log probabilities from the current policy pi_theta.
        old_log_probs: Log probabilities from the old policy pi_old.
        beta: Penalty coefficient.
        delta: KL divergence threshold.
        action_mask: Mask for valid actions.
        """
        # KL divergence: E_pi_old[log(pi_old / pi_theta)] = E_pi_old[old_log_probs - log_probs]
        # We approximate the expectation with the sample mean.
        # Note: The order is important for the gradient calculation.
        # log_ratio = old_log_probs - log_probs
        # kl_div_samples = log_ratio
        # kl_div = masked_mean(kl_div_samples, action_mask, dim=None) # element-wise mean

        # Per-sequence KL: sum_t (old_log_probs_t - log_probs_t)
        # Average KL over batch
        kl_div_per_sequence = old_log_probs - log_probs  # D_KL(pi_old || pi_theta) for each token
        if action_mask is not None:
            kl_div = masked_mean(kl_div_per_sequence, action_mask, dim=None)
            # Sum over seq dim, then mean over batch dim if action_mask is per token
            # kl_div_sequences = (kl_div_per_sequence * action_mask).sum(dim=-1) / action_mask.sum(dim=-1).clamp(min=1)
            # kl_div = kl_div_sequences.mean()

        else:
            # kl_div_sequences = kl_div_per_sequence.sum(dim=-1)
            # kl_div = kl_div_sequences.mean()
            kl_div = kl_div_per_sequence.mean()

        # Penalty term: beta * [KL - delta]_+^2
        # The gradient of the penalty term is 2 * beta * [KL - delta]_+ * nabla(KL)
        # Since KL = mean(old_log_probs - log_probs), nabla(KL) w.r.t log_probs is -1/N
        # The loss is J - penalty, so we subtract the penalty.
        # For gradient ascent, we want grad(J) - grad(penalty).
        # grad(penalty) = 2 * beta * max(0, kl_div - delta) * (-1/N) with respect to log_probs(theta)
        # So, -grad(penalty) = 2 * beta * max(0, kl_div - delta) * (1/N)
        # This will be handled by autograd. We just need to compute the penalty value.
        penalty_factor = torch.relu(kl_div - delta)
        penalty = beta * (penalty_factor**2)
        return penalty, kl_div.detach()  # Detach KL for stats, penalty has grads


class DisCOBasicLoss(nn.Module):
    """
    DisCO-b Loss: J1 objective - KL penalty
    J1 = E_{q ~ D, o ~ pi_old(·|q), r(o,q)=1} [s_theta(o,q)] - E_{q ~ D, o' ~ pi_old(·|q), r(o',q)=0} [s_theta(o',q)]
    Assuming l(s) = s for the difference.
    """

    def __init__(self, beta: float, delta: float, disco_scoring_func: str = "log_l"):
        super().__init__()
        self.beta = beta
        self.delta = delta
        self.disco_scoring_func = disco_scoring_func

    def forward(
        self,
        log_probs: torch.Tensor,  # log_probs from current policy pi_theta(o|q)
        old_log_probs: torch.Tensor,  # log_probs from old policy pi_old(o|q)
        rewards: torch.Tensor,  # Binary rewards (1 for positive, 0 for negative)
        action_mask: Optional[torch.Tensor] = None,  # Mask for valid actions in sequences
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        log_probs: (batch_size, seq_len)
        old_log_probs: (batch_size, seq_len)
        rewards: (batch_size,) indicating positive (1) or negative (0) samples
        action_mask: (batch_size, seq_len)
        """
        scores = DisCOHelper.calculate_scores(log_probs, old_log_probs, self.disco_scoring_func, action_mask)
        # scores shape: (batch_size,)

        positive_mask = rewards == 1
        negative_mask = rewards == 0

        # Ensure there are positive and negative samples to avoid division by zero or NaN
        if not positive_mask.any() or not negative_mask.any():
            # If one set is empty, J1 objective is ill-defined or zero.
            # Return zero objective and only KL penalty.
            # Or, if only one type exists, the objective might be considered as just that part.
            # For simplicity, if we don't have both, let objective be 0.
            # This case should ideally be handled by the sampler ensuring diverse batches.
            j1_objective = torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
            # Still calculate penalty if possible
            penalty, kl_div = DisCOHelper.calculate_kl_penalty(
                log_probs, old_log_probs, self.beta, self.delta, action_mask
            )
            # Loss = Objective - Penalty. For maximization, we negate this: Penalty - Objective
            loss = penalty - j1_objective  # Negative sign because we want to maximize J1
            return loss, kl_div, j1_objective.detach(), penalty.detach()

        s_positive = scores[positive_mask]
        s_negative = scores[negative_mask]

        # J1 = E[s_theta(o,q) | r=1] - E[s_theta(o',q) | r=0]
        # Approximated by mean of scores for positive and negative samples
        mean_s_positive = s_positive.mean() if s_positive.numel() > 0 else torch.tensor(0.0, device=scores.device)
        mean_s_negative = s_negative.mean() if s_negative.numel() > 0 else torch.tensor(0.0, device=scores.device)

        j1_objective = mean_s_positive - mean_s_negative

        # KL divergence penalty
        penalty, kl_div = DisCOHelper.calculate_kl_penalty(
            log_probs, old_log_probs, self.beta, self.delta, action_mask
        )

        # The paper aims to MAXIMIZE J - penalty.
        # Standard PyTorch optimizers MINIMIZE loss. So, loss = -(J - penalty) = penalty - J.
        loss = penalty - j1_objective

        return loss, kl_div.detach(), j1_objective.detach(), penalty.detach()


class DisCOLoss(nn.Module):
    """
    DisCO Loss (DRO): J2 objective - KL penalty
    J2 = E_{q ~ D, o ~ pi_old(·|q), r(o,q)=1} [s_theta(o,q)] - tau * log E_{q ~ D, o' ~ pi_old(·|q), r(o',q)=0} [exp(s_theta(o',q)/tau)]
    """

    def __init__(self, beta: float, delta: float, tau: float, disco_scoring_func: str = "log_l"):
        super().__init__()
        self.beta = beta
        self.delta = delta
        self.tau = tau
        self.disco_scoring_func = disco_scoring_func
        if self.tau <= 0:
            raise ValueError("tau must be positive for DisCOLoss.")

    def forward(
        self,
        log_probs: torch.Tensor,  # log_probs from current policy pi_theta(o|q)
        old_log_probs: torch.Tensor,  # log_probs from old policy pi_old(o|q)
        rewards: torch.Tensor,  # Binary rewards (1 for positive, 0 for negative)
        action_mask: Optional[torch.Tensor] = None,  # Mask for valid actions in sequences
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        log_probs: (batch_size, seq_len)
        old_log_probs: (batch_size, seq_len)
        rewards: (batch_size,) indicating positive (1) or negative (0) samples
        action_mask: (batch_size, seq_len)
        """
        scores = DisCOHelper.calculate_scores(log_probs, old_log_probs, self.disco_scoring_func, action_mask)
        # scores shape: (batch_size,)

        positive_mask = rewards == 1
        negative_mask = rewards == 0

        # Ensure there are positive and negative samples
        if not positive_mask.any() or not negative_mask.any():
            # If one set is empty, J2 objective is ill-defined or zero.
            j2_objective = torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
            penalty, kl_div = DisCOHelper.calculate_kl_penalty(
                log_probs, old_log_probs, self.beta, self.delta, action_mask
            )
            loss = penalty - j2_objective
            return loss, kl_div, j2_objective.detach(), penalty.detach()

        s_positive = scores[positive_mask]
        s_negative = scores[negative_mask]

        # Mean of scores for positive samples
        mean_s_positive = s_positive.mean() if s_positive.numel() > 0 else torch.tensor(0.0, device=scores.device)

        # For negative samples: tau * log E [exp(s_theta(o',q)/tau)]
        # Approximated by tau * logmeanexp(s_negative / tau)
        # logmeanexp(x) = log( (1/N) * sum(exp(x_i)) ) = logsumexp(x) - log(N)
        if s_negative.numel() > 0:
            # logsumexp provides numerical stability
            dro_term_negative = self.tau * (
                torch.logsumexp(s_negative / self.tau, dim=0)
                - torch.log(
                    torch.tensor(
                        s_negative.numel(),
                        device=scores.device,
                        dtype=self.tau.dtype if isinstance(self.tau, torch.Tensor) else torch.float,
                    )
                )
            )
        else:
            dro_term_negative = torch.tensor(0.0, device=scores.device, dtype=scores.dtype)

        j2_objective = mean_s_positive - dro_term_negative

        # KL divergence penalty
        penalty, kl_div = DisCOHelper.calculate_kl_penalty(
            log_probs, old_log_probs, self.beta, self.delta, action_mask
        )

        # Maximize J2 - penalty => Minimize penalty - J2
        loss = penalty - j2_objective

        return loss, kl_div.detach(), j2_objective.detach(), penalty.detach()
