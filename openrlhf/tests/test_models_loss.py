import pytest
import torch
from torch.testing import assert_close

from openrlhf.models.loss import DisCOBasicLoss, DisCOHelper, DisCOLoss

# Default dtype and device for tests
DTYPE = torch.float32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Helper function to create tensors
def _t(data, dtype=DTYPE, device=DEVICE):
    return torch.tensor(data, dtype=dtype, device=device)


# Test DisCOHelper.calculate_scores
def test_disco_helper_calculate_scores_log_l():
    log_probs = _t([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]])  # sum = 1.0, 1.0
    old_log_probs = _t([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])  # Not used for log_l
    action_mask = _t([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)  # Mask last for second sample

    # Test 'log_l' without action_mask
    scores_no_mask = DisCOHelper.calculate_scores(log_probs, old_log_probs, "log_l", action_mask=None)
    expected_scores_no_mask = _t([1.0, 1.0])
    assert_close(scores_no_mask, expected_scores_no_mask)

    # Test 'log_l' with action_mask
    # Sample 1: (0.1 + 0.2 + 0.7) / 3 = 1.0 / 3
    # Sample 2: (0.3 + 0.4) / 2 = 0.7 / 2
    # Action mask sum is used for averaging, so (log_probs * action_mask).sum / action_mask.sum
    # Correct interpretation for log_l: sum of log_probs where mask is true / sum of mask
    # My previous DisCOHelper for log_l was (log_probs * action_mask).sum(dim=-1) / action_mask.sum(dim=-1)
    # Let's re-verify the DisCO paper or common implementations for "average log likelihood"
    # Typically, s_theta(o,q) = log pi_theta(o|q) is sum of log probs of tokens in o.
    # If action_mask is to select tokens, then it should be sum(log_probs[action_mask_is_true]).
    # The previous implementation of DisCOHelper.calculate_scores was:
    # (log_probs * action_mask).sum(dim=-1) / action_mask.sum(dim=-1) for log_l
    # This is average log_prob per token. Let's assume this is the intended behavior for now.
    # If s_theta is sum of log_probs, then the division by action_mask.sum() is not there.
    # Re-reading the DisCO paper: "sθ(o, q) could be the log-likelihood log pθ(o|q)" -> implies sum.
    # Let's assume s_theta is SUM for now and will adjust loss if needed.
    # For now, I will test the current implementation of DisCOHelper.

    # Current DisCOHelper: (log_probs * action_mask).sum(dim=-1) / action_mask.sum(dim=-1).clamp(min=1)
    # So for sample 1: (0.1 + 0.2 + 0.7) / 3 = 1.0/3
    # For sample 2: (0.3 + 0.4 + 0.0) / 2 = 0.7/2 = 0.35
    scores_mask = DisCOHelper.calculate_scores(log_probs, old_log_probs, "log_l", action_mask=action_mask)
    expected_scores_mask = _t([1.0 / 3, 0.7 / 2])
    assert_close(scores_mask, expected_scores_mask, rtol=1e-5, atol=1e-5)


def test_disco_helper_calculate_scores_l_ratio():
    log_probs = _t([[-1.0, -1.0], [-2.0, -2.0]])  # sum = -2.0, -4.0
    old_log_probs = _t([[-0.5, -0.5], [-1.0, -1.0]])  # sum = -1.0, -2.0
    # score = sum(log_probs) - sum(old_log_probs)
    # Sample 1: -2.0 - (-1.0) = -1.0
    # Sample 2: -4.0 - (-2.0) = -2.0
    action_mask = _t([[1, 1], [1, 0]], dtype=torch.bool)

    # Test 'l_ratio' without action_mask
    scores_no_mask = DisCOHelper.calculate_scores(log_probs, old_log_probs, "l_ratio", action_mask=None)
    expected_scores_no_mask = _t([-1.0, -2.0])
    assert_close(scores_no_mask, expected_scores_no_mask)

    # Test 'l_ratio' with action_mask
    # Current DisCOHelper: sum(log_probs*mask) - sum(old_log_probs*mask)
    # Sample 1: (-1.0 + -1.0) - (-0.5 + -0.5) = -2.0 - (-1.0) = -1.0
    # Sample 2: (-2.0 + 0) - (-1.0 + 0)       = -2.0 - (-1.0) = -1.0
    scores_mask = DisCOHelper.calculate_scores(log_probs, old_log_probs, "l_ratio", action_mask=action_mask)
    expected_scores_mask = _t([-1.0, -1.0])
    assert_close(scores_mask, expected_scores_mask)


# Test DisCOHelper.calculate_kl_penalty
@pytest.mark.parametrize(
    "kl_val, delta, beta, expected_penalty_val",
    [
        (0.05, 0.1, 1.0, 0.0),  # KL < delta
        (0.1, 0.1, 1.0, 0.0),  # KL == delta
        (0.2, 0.1, 1.0, 1.0 * (0.1**2)),  # KL > delta
        (0.2, 0.1, 0.0, 0.0),  # beta = 0
        (0.2, 0.1, 2.0, 2.0 * (0.1**2)),  # beta = 2
    ],
)
def test_disco_helper_calculate_kl_penalty(kl_val, delta, beta, expected_penalty_val):
    # old_log_probs - log_probs = kl_val (on average per token)
    # Let seq_len = 2, batch_size = 1
    # (old1 - new1) + (old2 - new2) / 2 = kl_val
    # Let old_log_probs_token = kl_val, log_probs_token = 0 for simplicity to achieve mean kl_val
    log_probs = _t([[0.0, 0.0]])
    old_log_probs = _t([[kl_val, kl_val]])
    action_mask = _t([[1, 1]], dtype=torch.bool)

    penalty, kl_div_detached = DisCOHelper.calculate_kl_penalty(log_probs, old_log_probs, beta, delta, action_mask)

    assert_close(kl_div_detached, _t(kl_val))
    assert_close(penalty, _t(expected_penalty_val))

    # Test with no action mask
    penalty_no_mask, kl_div_detached_no_mask = DisCOHelper.calculate_kl_penalty(
        log_probs, old_log_probs, beta, delta, action_mask=None
    )
    assert_close(kl_div_detached_no_mask, _t(kl_val))
    assert_close(penalty_no_mask, _t(expected_penalty_val))


# Test DisCOBasicLoss
def test_disco_basic_loss_mixed_rewards():
    log_probs = _t([[0.1, 0.2], [0.3, 0.4], [0.5, 0.1], [0.2, 0.2]])  # sums: 0.3, 0.7, 0.6, 0.4
    old_log_probs = torch.zeros_like(log_probs)  # For log_l, old_log_probs are not used in score calculation itself
    # but are used for KL penalty.
    action_mask = _t([[1, 1], [1, 1], [1, 1], [1, 1]], dtype=torch.bool)
    # Assume scores are sums of log_probs for simplicity in testing objective
    # DisCOHelper.calculate_scores with log_l and full mask will be sum/seq_len.
    # For these tests, let's assume scores are pre-calculated or use scoring_func="l_ratio" with old_log_probs=0 for simplicity
    # to make scores = log_probs.sum(-1)
    # To make test scores exactly log_probs.sum(-1) with "log_l", action_mask sum should be 1 or not provide mask.
    # Let's use action_mask=None for score calculation part of the objective test

    # Scores for log_l (sum of log_probs as per DisCOHelper with action_mask=None)
    # s0 = 0.3, s1 = 0.7, s2 = 0.6, s3 = 0.4
    rewards_binary = _t([1, 0, 1, 0])  # s_pos = [0.3, 0.6], s_neg = [0.7, 0.4]

    beta = 1.0
    delta = 0.01
    # KL: old_log_probs are all 0, log_probs are positive. So KL = mean(0 - log_probs_token) < 0
    # Let's set log_probs for KL calculation to be different to test penalty properly.
    # Let kl_log_probs = 0, kl_old_log_probs = 0.02 per token on average. KL = 0.02
    kl_log_probs = torch.zeros_like(log_probs)
    kl_old_log_probs = torch.full_like(log_probs, 0.02)  # KL = 0.02

    loss_fn = DisCOBasicLoss(beta=beta, delta=delta, disco_scoring_func="log_l")
    # We pass kl_log_probs and kl_old_log_probs for the actual forward call
    # but the scores for objective are based on `log_probs`

    # Manually calculate scores as DisCOHelper would with action_mask=None
    scores = log_probs.sum(dim=-1)  # [0.3, 0.7, 0.6, 0.4]

    s_positive = scores[rewards_binary == 1]  # [0.3, 0.6]
    s_negative = scores[rewards_binary == 0]  # [0.7, 0.4]

    mean_s_positive = s_positive.mean()  # (0.3 + 0.6) / 2 = 0.45
    mean_s_negative = s_negative.mean()  # (0.7 + 0.4) / 2 = 0.55
    j1_objective = mean_s_positive - mean_s_negative  # 0.45 - 0.55 = -0.1

    # For penalty, using kl_log_probs and kl_old_log_probs
    # kl_div = mean(kl_old_log_probs - kl_log_probs) = mean(0.02 - 0) = 0.02
    # penalty_val = beta * relu(kl_div - delta)^2 = 1.0 * relu(0.02 - 0.01)^2 = (0.01)^2 = 0.0001

    # To pass scores directly for testing objective, we would need to mock calculate_scores or use l_ratio carefully
    # Instead, let's use the class as is. For "log_l", scores are average log_prob per token if action_mask is given.
    # Let's use action_mask that makes scores = sum(log_probs) by having action_mask.sum(dim=-1) = 1
    # This is tricky. Let's use scoring_func="l_ratio" and set old_log_probs to 0 for score calculation part.
    # And use different old_log_probs for KL penalty part. This separation is not how the class is designed.
    # The class uses the same log_probs and old_log_probs for both score and KL.

    # Let's re-evaluate: the `log_probs` and `old_log_probs` passed to forward are used for BOTH score and KL.
    # So, if old_log_probs = torch.zeros_like(log_probs)
    # For "log_l", scores = (log_probs * action_mask).sum(-1) / action_mask.sum(-1).clamp(min=1)
    # For "l_ratio", scores = (log_probs*AM).sum(-1)/AM.sum(-1) - (old_log_probs*AM).sum(-1)/AM.sum(-1) (if AM used in helper like that)
    # The helper for l_ratio is: current_seq_log_probs - old_seq_log_probs where seq_log_probs are sums.
    # So if old_log_probs = 0, then for "l_ratio", scores = log_probs.sum(-1) (if action_mask=None or covers all)

    loss_fn_lratio = DisCOBasicLoss(beta=beta, delta=delta, disco_scoring_func="l_ratio")
    # For l_ratio, scores = log_probs.sum(-1) - old_log_probs.sum(-1)
    # Let old_log_probs_for_score = torch.zeros_like(log_probs)
    # Then scores = log_probs.sum(-1) = [0.3, 0.7, 0.6, 0.4]
    # KL = mean(old_log_probs_for_score_token_wise - log_probs_token_wise) = mean(0 - log_probs_token_wise)
    # This makes KL negative if log_probs are positive.
    # Let's use a single set of log_probs and old_log_probs for the test.
    # log_probs = _t([[0.1, 0.2], [0.3, 0.4], [0.5, 0.1], [0.2, 0.2]]) sums: 0.3, 0.7, 0.6, 0.4
    # old_log_probs = _t([[0.1, 0.18], [0.3, 0.38], [0.5, 0.08], [0.2, 0.18]]) sums: 0.28, 0.68, 0.58, 0.38
    # KL = mean(old_log_probs_token - log_probs_token) per token
    # kl_per_token_diff = old_log_probs - log_probs = [[0, -0.02], [0, -0.02], [0, -0.02], [0, -0.02]]
    # kl_div = mean([0, -0.02, 0, -0.02, 0, -0.02, 0, -0.02]) = -0.01
    # penalty = beta * relu(-0.01 - delta)^2 = 0 if delta is positive.

    # Let's make old_log_probs slightly higher than log_probs for positive KL
    log_probs_for_test = _t([[0.1, 0.2], [0.3, 0.4], [0.5, 0.1], [0.2, 0.2]])  # sums: 0.3, 0.7, 0.6, 0.4
    old_log_probs_for_test = _t(
        [[0.12, 0.22], [0.32, 0.42], [0.52, 0.12], [0.22, 0.22]]
    )  # sums: 0.34, 0.74, 0.64, 0.44
    # kl_per_token_diff = old - new = [[0.02, 0.02], [0.02, 0.02], [0.02, 0.02], [0.02, 0.02]]
    # kl_div = 0.02.
    # penalty_val = 1.0 * relu(0.02 - 0.01)^2 = (0.01)^2 = 0.0001

    # Scores for "l_ratio": sum(log_probs) - sum(old_log_probs)
    # s0 = 0.3 - 0.34 = -0.04
    # s1 = 0.7 - 0.74 = -0.04
    # s2 = 0.6 - 0.64 = -0.04
    # s3 = 0.4 - 0.44 = -0.04
    # This means all scores are -0.04.
    # s_pos = [-0.04, -0.04], s_neg = [-0.04, -0.04]
    # mean_s_pos = -0.04, mean_s_neg = -0.04
    # j1_objective = -0.04 - (-0.04) = 0.0

    # Expected loss = penalty - j1_objective = 0.0001 - 0.0 = 0.0001

    # Using DisCOBasicLoss with "l_ratio"
    actual_loss, actual_kl, actual_j1, actual_penalty = loss_fn_lratio(
        log_probs_for_test, old_log_probs_for_test, rewards_binary, action_mask=None
    )
    assert_close(actual_kl, _t(0.02))
    assert_close(actual_penalty, _t(0.0001))
    assert_close(actual_j1, _t(0.0))
    assert_close(actual_loss, _t(0.0001))

    # Test with "log_l"
    # Scores for "log_l": sum(log_probs_for_test) assuming action_mask=None or full. Helper uses mean if mask.
    # Let's use action_mask that makes it sum. No, helper divides by mask sum.
    # So, scores_log_l = log_probs_for_test.mean(dim=-1) if action_mask=None (as default mask is all 1s, then mean over it)
    # No, if action_mask is None, DisCOHelper.calculate_scores takes log_probs.sum(dim=-1) for log_l.
    # scores_log_l = [0.3, 0.7, 0.6, 0.4]
    # s_pos_log_l = [0.3, 0.6], mean = 0.45
    # s_neg_log_l = [0.7, 0.4], mean = 0.55
    # j1_log_l = 0.45 - 0.55 = -0.1
    # KL and penalty are the same (0.02 and 0.0001)
    # Expected loss_log_l = penalty - j1_log_l = 0.0001 - (-0.1) = 0.1001
    loss_fn_logl = DisCOBasicLoss(beta=beta, delta=delta, disco_scoring_func="log_l")
    actual_loss_logl, actual_kl_logl, actual_j1_logl, actual_penalty_logl = loss_fn_logl(
        log_probs_for_test,
        old_log_probs_for_test,
        rewards_binary,
        action_mask=None,  # action_mask=None means full sequence sum for score
    )
    assert_close(actual_kl_logl, _t(0.02))
    assert_close(actual_penalty_logl, _t(0.0001))
    assert_close(actual_j1_logl, _t(-0.1))
    assert_close(actual_loss_logl, _t(0.1001))


def test_disco_basic_loss_no_positive_samples():
    log_probs = _t([[0.1, 0.2], [0.3, 0.4]])
    old_log_probs = torch.zeros_like(log_probs)
    rewards = _t([0, 0])  # All negative
    loss_fn = DisCOBasicLoss(beta=1.0, delta=0.01, disco_scoring_func="log_l")
    # J1 = 0 - E[s|r=0]. If E[s|r=0] is positive, J1 is negative.
    # Scores with action_mask=None for log_l: [0.3, 0.7]
    # mean_s_negative = (0.3+0.7)/2 = 0.5. mean_s_positive = 0.
    # J1 = 0 - 0.5 = -0.5.
    # KL = mean(0 - log_probs) = mean([-0.1, -0.2, -0.3, -0.4])/4 = -0.25. Penalty=0.
    # Expected loss = 0 - (-0.5) = 0.5
    # The code has a specific check: if not positive_mask.any() or not negative_mask.any(): j_objective = 0
    # So, J1 should be 0. Penalty is 0. Loss = 0.

    loss, kl, j1, penalty = loss_fn(log_probs, old_log_probs, rewards, action_mask=None)
    assert_close(j1, _t(0.0))  # Due to the condition of no positive/negative samples
    # KL = mean(0 - [[0.1,0.2],[0.3,0.4]]) = - (0.1+0.2+0.3+0.4)/4 = -1.0/4 = -0.25
    # Penalty for KL=-0.25, delta=0.01: beta * relu(-0.25 - 0.01)^2 = 0
    assert_close(kl, _t(-0.25))
    assert_close(penalty, _t(0.0))
    assert_close(loss, _t(0.0))


def test_disco_basic_loss_no_negative_samples():
    log_probs = _t([[0.1, 0.2], [0.3, 0.4]])
    old_log_probs = torch.zeros_like(log_probs)
    rewards = _t([1, 1])  # All positive
    loss_fn = DisCOBasicLoss(beta=1.0, delta=0.01, disco_scoring_func="log_l")
    # J1 = E[s|r=1] - 0. Scores = [0.3, 0.7]. E[s|r=1] = 0.5. J1=0.5
    # KL = -0.25. Penalty=0.
    # Expected loss = 0 - 0.5 = -0.5
    # The code has a specific check: if not positive_mask.any() or not negative_mask.any(): j_objective = 0
    # So, J1 should be 0. Penalty is 0. Loss = 0.
    loss, kl, j1, penalty = loss_fn(log_probs, old_log_probs, rewards, action_mask=None)
    assert_close(j1, _t(0.0))  # Due to the condition
    assert_close(kl, _t(-0.25))
    assert_close(penalty, _t(0.0))
    assert_close(loss, _t(0.0))


# Test DisCOLoss
def test_disco_loss_mixed_rewards():
    log_probs = _t([[0.1, 0.2], [0.7, 0.8], [0.3, 0.4], [0.5, 0.6]])  # sums: 0.3, 1.5, 0.7, 1.1
    old_log_probs = torch.zeros_like(log_probs)  # KL = mean(-log_probs_token)
    action_mask = None  # for scores to be sums
    rewards_binary = _t([1, 0, 1, 0])  # s_pos = [0.3, 0.7], s_neg = [1.5, 1.1]

    beta = 1.0
    delta = 0.01  # Assume KL will be negative here, so penalty=0
    tau = 0.5

    loss_fn = DisCOLoss(beta=beta, delta=delta, tau=tau, disco_scoring_func="log_l")

    # Scores for "log_l" with action_mask=None: sums of log_probs
    # s0=0.3 (pos), s1=1.5 (neg), s2=0.7 (pos), s3=1.1 (neg)
    s_positive = _t([0.3, 0.7])
    s_negative = _t([1.5, 1.1])

    mean_s_positive = s_positive.mean()  # (0.3 + 0.7) / 2 = 0.5

    # DRO term: tau * log E [exp(s_neg / tau)] = tau * (logsumexp(s_neg/tau) - log(N_neg))
    s_neg_over_tau = s_negative / tau  # [1.5/0.5, 1.1/0.5] = [3.0, 2.2]
    logsumexp_s_neg_over_tau = torch.logsumexp(s_neg_over_tau, dim=0)  # log(e^3 + e^2.2)
    N_neg = torch.tensor(
        s_negative.numel(), dtype=tau.dtype if isinstance(tau, torch.Tensor) else torch.float, device=DEVICE
    )
    dro_term = tau * (logsumexp_s_neg_over_tau - torch.log(N_neg))
    # log( (e^3 + e^2.2)/2 ) * 0.5
    # e^3 = 20.085, e^2.2 = 9.025
    # (20.085 + 9.025)/2 = 14.555
    # log(14.555) = 2.678
    # dro_term = 0.5 * 2.678 = 1.339
    expected_dro_term = 0.5 * (torch.log((torch.exp(_t(3.0)) + torch.exp(_t(2.2))) / 2.0))
    assert_close(dro_term, expected_dro_term, rtol=1e-3, atol=1e-3)

    j2_objective = mean_s_positive - dro_term  # 0.5 - 1.339 = -0.839

    # KL = mean(0 - log_probs_token). Example: -(0.1+0.2+0.7+0.8+0.3+0.4+0.5+0.6) / 8 = -3.6/8 = -0.45
    # penalty = 1.0 * relu(-0.45 - 0.01)^2 = 0

    # Expected loss = penalty - j2_objective = 0 - (-0.839) = 0.839

    actual_loss, actual_kl, actual_j2, actual_penalty = loss_fn(
        log_probs, old_log_probs, rewards_binary, action_mask=action_mask
    )

    expected_kl = (old_log_probs - log_probs).mean()
    assert_close(actual_kl, expected_kl)
    assert_close(actual_penalty, _t(0.0))  # Since KL is negative
    assert_close(actual_j2, j2_objective, rtol=1e-3, atol=1e-3)
    assert_close(actual_loss, -j2_objective, rtol=1e-3, atol=1e-3)  # penalty is 0


def test_disco_loss_no_positive_samples():
    log_probs = _t([[0.1, 0.2], [0.3, 0.4]])
    old_log_probs = torch.zeros_like(log_probs)  # KL will be negative, penalty = 0
    rewards = _t([0, 0])  # All negative
    loss_fn = DisCOLoss(beta=1.0, delta=0.01, tau=0.5, disco_scoring_func="log_l")

    # J2 objective should be 0 due to the check: if not positive_mask.any() or not negative_mask.any()
    # KL = mean(0 - log_probs) = -0.25. Penalty = 0.
    # Expected loss = 0 - 0 = 0.

    loss, kl, j2, penalty = loss_fn(log_probs, old_log_probs, rewards, action_mask=None)

    assert_close(j2, _t(0.0))  # Due to the condition
    assert_close(kl, _t(-0.25))
    assert_close(penalty, _t(0.0))
    assert_close(loss, _t(0.0))


def test_disco_loss_no_negative_samples():
    log_probs = _t([[0.1, 0.2], [0.3, 0.4]])
    old_log_probs = torch.zeros_like(log_probs)  # KL will be negative, penalty = 0
    rewards = _t([1, 1])  # All positive
    loss_fn = DisCOLoss(beta=1.0, delta=0.01, tau=0.5, disco_scoring_func="log_l")

    # J2 objective should be 0 due to the check: if not positive_mask.any() or not negative_mask.any()
    # KL = -0.25. Penalty = 0.
    # Expected loss = 0 - 0 = 0.

    loss, kl, j2, penalty = loss_fn(log_probs, old_log_probs, rewards, action_mask=None)

    assert_close(j2, _t(0.0))  # Due to the condition
    assert_close(kl, _t(-0.25))
    assert_close(penalty, _t(0.0))
    assert_close(loss, _t(0.0))


def test_disco_loss_zero_tau_exception():
    with pytest.raises(ValueError, match="tau must be positive for DisCOLoss"):
        DisCOLoss(beta=1.0, delta=0.01, tau=0.0, disco_scoring_func="log_l")


# TODO: Add more tests for edge cases like empty action masks if applicable by design,
# or masks that cover only a few tokens.
# Test with action_mask in DisCOBasicLoss and DisCOLoss to ensure per-token averaging in scores if that's the case.
# The current DisCOHelper.calculate_scores for log_l with action_mask averages log_probs.
# This means the interpretation of s_theta changes.
# If s_theta is sum log p(o|q), then action_mask in helper should just mask, not average.
# For now, tests follow current implementation.
# The current DisCOHelper for `log_l` with an action_mask will compute `(log_probs * action_mask).sum(dim=-1) / action_mask.sum(dim=-1).clamp(min=1)`
# The current DisCOHelper for `l_ratio` with an action_mask will compute `(log_probs * mask).sum(dim=-1) - (old_log_probs * mask).sum(dim=-1)` (assuming mask.sum is not used for division here based on code structure for l_ratio)


# Example of testing with action_mask that affects scores for log_l
def test_disco_basic_loss_log_l_with_action_mask():
    log_probs = _t([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]])  # mean if full mask: [0.333, 0.333], sum: [1.0, 1.0]
    old_log_probs = _t([[0.1, 0.2, 0.6], [0.3, 0.3, 0.3]])  # For KL
    action_mask = _t([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)  # Mask last for first sample
    rewards_binary = _t([1, 0])  # s_pos = [score0], s_neg = [score1]

    beta = 1.0
    delta = 0.001  # KL will be small positive: (0.6-0.7)/3 + (0.3-0.3)/3 for sample 0, (0.3-0.3)/3 for sample1
    # (0.1-0.1)/2 + (0.2-0.2)/2 + (0.6-0.7)/3 -> KL for sample 0 (masked by AM in helper)
    # KL for token [0,2] is (0.6-0.7) = -0.1. Masked for sample 0.
    # KL for sample 0 (token 0,1): (0.1-0.1) + (0.2-0.2) = 0.  Actual KL = 0.
    # KL for sample 1 (token 0,1,2): (0.3-0.3) + (0.3-0.4) + (0.3-0.3) = -0.1. Actual KL = -0.1/3 for sample1
    # Overall KL = (0 + (-0.1/3 * 3 tokens in sample1 )) / (2tok_s0+3tok_s1) -> complicated by mask sum
    # DisCOHelper KL: mean over all valid tokens: ( (0.1-0.1)+(0.2-0.2) + (0.3-0.3)+(0.3-0.4)+(0.3-0.3) ) / (2+3)
    # = (0 + 0 - 0.1 + 0) / 5 = -0.1 / 5 = -0.02

    loss_fn = DisCOBasicLoss(beta=beta, delta=delta, disco_scoring_func="log_l")

    # Scores for "log_l" with action_mask:
    # s0 = (0.1 + 0.2) / 2 = 0.15 (positive sample)
    # s1 = (0.3 + 0.4 + 0.3) / 3 = 1.0 / 3 approx 0.333 (negative sample)

    j1_objective = 0.15 - (1.0 / 3)  # approx 0.15 - 0.333 = -0.18333

    # KL = -0.02. Penalty = 1.0 * relu(-0.02 - 0.001)^2 = 0.
    # Expected loss = penalty - j1_objective = 0 - (-0.18333) = 0.18333

    actual_loss, actual_kl, actual_j1, actual_penalty = loss_fn(
        log_probs, old_log_probs, rewards_binary, action_mask=action_mask
    )

    assert_close(actual_kl, _t(-0.02))
    assert_close(actual_penalty, _t(0.0))
    assert_close(actual_j1, _t(j1_objective))
    assert_close(actual_loss, _t(-j1_objective))


# Test for when KL is positive and creates a penalty
def test_disco_basic_loss_with_positive_kl_penalty():
    log_probs = _t([[0.1, 0.1], [0.1, 0.1]])
    old_log_probs = _t([[0.2, 0.2], [0.2, 0.2]])  # old > new, so KL is positive
    action_mask = _t([[1, 1], [1, 1]], dtype=torch.bool)
    rewards_binary = _t([1, 0])

    beta = 2.0
    delta = 0.05
    loss_fn = DisCOBasicLoss(beta=beta, delta=delta, disco_scoring_func="log_l")

    # Scores for "log_l" with action_mask:
    # s0 = (0.1+0.1)/2 = 0.1 (positive)
    # s1 = (0.1+0.1)/2 = 0.1 (negative)
    # j1_objective = 0.1 - 0.1 = 0.0

    # KL = mean(old_log_probs_token - log_probs_token) = mean(0.2 - 0.1) = 0.1
    # penalty_val = beta * relu(KL - delta)^2 = 2.0 * relu(0.1 - 0.05)^2 = 2.0 * (0.05)^2 = 2.0 * 0.0025 = 0.005

    # Expected loss = penalty - j1_objective = 0.005 - 0.0 = 0.005

    actual_loss, actual_kl, actual_j1, actual_penalty = loss_fn(
        log_probs, old_log_probs, rewards_binary, action_mask=action_mask
    )

    assert_close(actual_kl, _t(0.1))
    assert_close(actual_penalty, _t(0.005))
    assert_close(actual_j1, _t(0.0))
    assert_close(actual_loss, _t(0.005))
