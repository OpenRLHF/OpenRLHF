import importlib.util
import math
import sys
import types
from pathlib import Path

import pytest
import torch

_TEST_PACKAGE = "_openrlhf_loss_test"


def _load_loss_module():
    root = Path(__file__).resolve().parents[1]
    models_dir = root / "openrlhf" / "models"

    pkg = types.ModuleType(_TEST_PACKAGE)
    pkg.__path__ = [str(models_dir)]
    sys.modules[_TEST_PACKAGE] = pkg

    for name in ("utils", "loss"):
        spec = importlib.util.spec_from_file_location(f"{_TEST_PACKAGE}.{name}", models_dir / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"{_TEST_PACKAGE}.{name}"] = module
        spec.loader.exec_module(module)

    return sys.modules[f"{_TEST_PACKAGE}.loss"]


def _load_loss_utils_module():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "openrlhf.utils.loss_utils", root / "openrlhf" / "utils" / "loss_utils.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["openrlhf.utils.loss_utils"] = module
    spec.loader.exec_module(module)
    return module


_loss_module = _load_loss_module()
_loss_utils_module = _load_loss_utils_module()
PolicyLoss = _loss_module.PolicyLoss
aggregate_loss = _loss_module.aggregate_loss
get_loss_batch_info = _loss_utils_module.get_loss_batch_info
iter_grad_accum_global_norm = _loss_utils_module.iter_grad_accum_global_norm


def test_token_mean_aggregation_matches_verl_dp_average():
    loss = torch.tensor([[1.0, 2.0, 0.0], [3.0, 4.0, 5.0], [7.0, 0.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [1.0, 0.0, 0.0]])

    global_token_mean = aggregate_loss(loss, mask)
    batch_num_tokens = mask.sum()

    rank0 = aggregate_loss(loss[:2], mask[:2], dp_size=2, batch_num_tokens=batch_num_tokens)
    rank1 = aggregate_loss(loss[2:], mask[2:], dp_size=2, batch_num_tokens=batch_num_tokens)

    # DeepSpeed/DDP averages gradients across DP ranks; verl compensates by multiplying by dp_size.
    assert torch.allclose((rank0 + rank1) / 2, global_token_mean)


def test_seq_mean_token_mean_aggregation_matches_verl_dp_average():
    loss = torch.tensor([[1.0, 3.0, 9.0], [2.0, 4.0, 0.0], [8.0, 0.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

    global_seq_mean = aggregate_loss(loss, mask, token_level_loss=False)
    global_batch_size = torch.tensor(3.0)

    rank0 = aggregate_loss(
        loss[:2],
        mask[:2],
        token_level_loss=False,
        dp_size=2,
        global_batch_size=global_batch_size,
    )
    rank1 = aggregate_loss(
        loss[2:],
        mask[2:],
        token_level_loss=False,
        dp_size=2,
        global_batch_size=global_batch_size,
    )

    assert torch.allclose((rank0 + rank1) / 2, global_seq_mean)


def test_loss_batch_info_excludes_fully_masked_sequences():
    loss = torch.tensor([[1.0, 3.0], [5.0, 7.0]])
    mask = torch.tensor([[1.0, 1.0], [0.0, 0.0]])

    info = get_loss_batch_info(strategy=object(), loss_mask=mask)
    seq_loss = aggregate_loss(loss, mask, token_level_loss=False, **info)

    assert info["global_batch_size"].item() == 1.0
    assert torch.allclose(seq_loss, torch.tensor(2.0))


def test_policy_loss_token_mean_aggregation_matches_full_batch():
    log_probs = torch.tensor([[-0.20, -0.40, -0.10], [-0.70, -0.30, -0.20], [-0.60, -0.10, -0.50]])
    old_log_probs = log_probs.detach() - torch.tensor([[0.03, -0.04, 0.01], [0.02, 0.05, -0.03], [0.04, 0.0, -0.02]])
    advantages = torch.tensor([[1.0, -0.5, 0.0], [0.3, -1.2, 0.5], [1.5, 0.0, -0.7]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [1.0, 0.0, 1.0]])

    loss_fn = PolicyLoss(clip_eps_low=0.2, clip_eps_high=0.2)
    full_loss, *_ = loss_fn(log_probs, old_log_probs, advantages, action_mask=mask)
    batch_num_tokens = mask.sum()

    rank0_loss, *_ = loss_fn(
        log_probs[:2],
        old_log_probs[:2],
        advantages[:2],
        action_mask=mask[:2],
        dp_size=2,
        batch_num_tokens=batch_num_tokens,
    )
    rank1_loss, *_ = loss_fn(
        log_probs[2:],
        old_log_probs[2:],
        advantages[2:],
        action_mask=mask[2:],
        dp_size=2,
        batch_num_tokens=batch_num_tokens,
    )

    assert torch.allclose((rank0_loss + rank1_loss) / 2, full_loss)


def test_policy_kl_metric_uses_policy_ratio_when_vllm_correction_is_enabled():
    log_probs = torch.tensor([[-0.2, -0.4]])
    old_log_probs = torch.tensor([[-0.3, -0.1]])
    rollout_log_probs = torch.tensor([[-1.3, -1.1]])
    advantages = torch.ones_like(log_probs)
    mask = torch.ones_like(log_probs)

    loss_fn = PolicyLoss(is_correction_level="token", is_correction_mode="clip", is_correction_threshold=[0.1, 10.0])
    _, _, ppo_kl, vllm_kl, _ = loss_fn(
        log_probs,
        old_log_probs,
        advantages,
        action_mask=mask,
        rollout_log_probs=rollout_log_probs,
    )

    assert torch.allclose(ppo_kl, (old_log_probs - log_probs).mean())
    assert torch.allclose(vllm_kl, (rollout_log_probs - old_log_probs).mean())


def test_icepop_filters_overflowing_ratio_without_nan():
    log_probs = torch.zeros((1, 2), requires_grad=True)
    loss_fn = PolicyLoss(is_correction_level="token", is_correction_mode="mask", is_correction_threshold=[0.5, 5.0])
    loss, *_ = loss_fn(
        log_probs,
        torch.zeros_like(log_probs),
        torch.ones_like(log_probs),
        action_mask=torch.ones_like(log_probs),
        rollout_log_probs=torch.tensor([[-1000.0, 0.0]]),
    )
    loss.backward()

    assert torch.allclose(loss, torch.tensor(-0.5))
    assert torch.isfinite(log_probs.grad).all()
    assert log_probs.grad[0, 0] == 0


def test_grad_accum_global_norm_matches_global_token_mean_over_window():
    # Two micro-batches in one optimizer-step window (gas=2) with UNEVEN token counts.
    mb0 = torch.tensor([[1.0, 2.0, 0.0]])  # 2 tokens
    mask0 = torch.tensor([[1.0, 1.0, 0.0]])
    mb1 = torch.tensor([[3.0, 4.0, 5.0], [6.0, 7.0, 0.0]])  # 5 tokens
    mask1 = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])
    gas = 2

    # Reference: a single per-token mean over the whole optimizer step (global token count).
    n_global = mask0.sum() + mask1.sum()  # 7
    expected = ((mb0 * mask0).sum() + (mb1 * mask1).sum()) / n_global

    # Emulate the grad-accum path: each micro-batch is normalized by the SAME window-global
    # count (folded by gas via the helper), and DeepSpeed scales each backward loss by 1/gas;
    # the accumulated gradient is proportional to the sum of the scaled losses.
    items = [(mb0, mask0), (mb1, mask1)]
    masks = {id(mb0): mask0, id(mb1): mask1}
    emitted = list(
        iter_grad_accum_global_norm(
            items, strategy=object(), accumulated_gradient=gas, mask_fn=lambda it: masks[id(it[0])]
        )
    )
    assert len(emitted) == 2
    accumulated = 0.0
    for (loss_mat, loss_mask), info in emitted:
        # window-global count, folded by gas
        assert torch.allclose(info["batch_num_tokens"], n_global / gas)
        per_mb = aggregate_loss(loss_mat, loss_mask, **{k: v for k, v in info.items() if k != "global_batch_size"})
        accumulated = accumulated + per_mb / gas  # DeepSpeed _scale_loss_by_gas
    assert torch.allclose(accumulated, expected)


def test_grad_accum_global_norm_differs_from_per_microbatch_when_uneven():
    # Sanity: the OLD per-micro-batch normalization (mean of per-mb token-means) does NOT
    # equal the global token-mean when token counts are uneven -> confirms the fix matters.
    mb0 = torch.tensor([[1.0, 2.0, 0.0]])
    mask0 = torch.tensor([[1.0, 1.0, 0.0]])
    mb1 = torch.tensor([[3.0, 4.0, 5.0], [6.0, 7.0, 0.0]])
    mask1 = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])

    global_mean = ((mb0 * mask0).sum() + (mb1 * mask1).sum()) / (mask0.sum() + mask1.sum())
    per_mb = (aggregate_loss(mb0, mask0) + aggregate_loss(mb1, mask1)) / 2
    assert not torch.allclose(per_mb, global_mean)


def test_policy_kl_metric_is_not_clamped():
    log_probs = torch.tensor([[100.0]])
    old_log_probs = torch.zeros_like(log_probs)
    advantages = torch.ones_like(log_probs)
    mask = torch.ones_like(log_probs)

    _, _, ppo_kl, _, _ = PolicyLoss()(log_probs, old_log_probs, advantages, action_mask=mask)

    assert torch.allclose(ppo_kl, (old_log_probs - log_probs).mean())


def test_seq_mask_gates_on_the_per_sequence_geometric_mean_ratio():
    # Row 0: token ratios e^1 and e^-1 (geometric mean 1) -> kept, each token weighted by its own ratio.
    # Row 1: geometric mean e^2 > 5 -> the whole sequence is dropped.
    logp = torch.zeros(2, 2)
    rollout = torch.tensor([[-1.0, 1.0], [-2.0, -2.0]])
    loss_fn = PolicyLoss(is_correction_level="seq", is_correction_threshold=[0.5, 5.0])
    loss, _, _, _, filt = loss_fn(
        logp, logp, torch.ones(2, 2), action_mask=torch.ones(2, 2), rollout_log_probs=rollout
    )
    torch.testing.assert_close(loss, -torch.tensor([1.0, -1.0]).exp().sum() / 4)
    torch.testing.assert_close(filt, torch.tensor(0.5))


def test_binary_kl_trust_region_masks_tokens_and_sequences():
    # On-policy PPO ratio (old == logp) so the surrogate is the REINFORCE gradient. Token 0 is far off
    # the behavior policy (mu=1/3, pi=1), token 1 matches it (mu=pi=1/2). Token-level binary_kl drops
    # token 0 and keeps token 1 with IS weight 1: loss = -(0 + 1) / 2. Seq-level (FlashREINFORCE) drops
    # the whole sequence on its mean KL.
    logp = torch.tensor([[0.0, math.log(0.5)]])
    mu = torch.tensor([[-math.log(3.0), math.log(0.5)]])
    adv, mask = torch.ones(1, 2), torch.ones(1, 2)
    tok = PolicyLoss(is_correction_level="token", is_correction_gating="binary_kl", is_correction_threshold=[0, 0.05])
    loss, _, _, _, filt = tok(logp, logp, adv, action_mask=mask, rollout_log_probs=mu)
    torch.testing.assert_close(loss, torch.tensor(-0.5))
    torch.testing.assert_close(filt, torch.tensor(0.5))
    seq = PolicyLoss(is_correction_level="seq", is_correction_gating="binary_kl", is_correction_threshold=[0, 3e-3])
    loss, _, _, _, filt = seq(logp, logp, adv, action_mask=mask, rollout_log_probs=mu)
    torch.testing.assert_close(loss, torch.tensor(0.0))
    torch.testing.assert_close(filt, torch.tensor(1.0))
    # Two-sided: pi << mu on the sampled token is dropped just the same.
    _, _, _, _, filt = seq(mu, mu, adv, action_mask=mask, rollout_log_probs=logp)
    torch.testing.assert_close(filt, torch.tensor(1.0))
    # A tiny mismatch (pi=0.5 vs mu=0.505, binary KL ~5e-5) stays inside delta=3e-3.
    close = torch.full((1, 2), math.log(0.5))
    _, _, _, _, filt = seq(close, close, adv, action_mask=mask, rollout_log_probs=torch.full((1, 2), math.log(0.505)))
    torch.testing.assert_close(filt, torch.tensor(0.0))


def test_tv_trust_region_masks_by_sampled_token_probability_gap():
    # |pi - mu| on the sampled token: 0.6 vs 0.5 -> 0.1 > 0.05 dropped; 0.52 vs 0.5 kept with IS weight 0.52/0.5.
    logp = torch.tensor([[math.log(0.6), math.log(0.52)]])
    mu = torch.full((1, 2), math.log(0.5))
    loss_fn = PolicyLoss(is_correction_level="token", is_correction_gating="tv", is_correction_threshold=[0, 0.05])
    loss, _, _, _, filt = loss_fn(logp, logp, torch.ones(1, 2), action_mask=torch.ones(1, 2), rollout_log_probs=mu)
    torch.testing.assert_close(filt, torch.tensor(0.5))
    torch.testing.assert_close(loss, torch.tensor(-(0.52 / 0.5) / 2))


def test_invalid_is_correction_combinations_are_rejected():
    for kw in (
        dict(is_correction_level="seq", is_correction_mode="clip"),
        dict(is_correction_level="token", is_correction_mode="clip", is_correction_gating="binary_kl"),
        dict(is_correction_level="token", is_correction_gating="kl"),
        dict(is_correction_level="geo"),
    ):
        with pytest.raises(ValueError):
            PolicyLoss(is_correction_threshold=[0, 0.05], **kw)
    with pytest.raises(ValueError, match="upper bound only"):  # the ratio band [0.5, 5] would reject everything
        PolicyLoss(is_correction_level="seq", is_correction_gating="binary_kl", is_correction_threshold=[0.5, 5.0])


def test_seq_mean_token_mean_weighs_every_sequence_the_same():
    # Two rows of unequal length: loss = -(A0 + A1) / 2, gradient -A_i / (T_i * B) per token.
    logp = torch.full((2, 2), -0.6931, requires_grad=True)
    mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    adv = torch.tensor([[2.0, 2.0], [1.0, 0.0]])
    loss, *_ = PolicyLoss(token_level_loss=False)(logp, logp.detach(), adv, action_mask=mask)
    torch.testing.assert_close(loss, torch.tensor(-1.5))
    loss.backward()
    torch.testing.assert_close(logp.grad, torch.tensor([[-0.5, -0.5], [-0.5, 0.0]]))


@pytest.mark.parametrize(
    "kind,token_level",
    [("sft", True), ("sft", False), ("value", True), ("value", False), ("ppo", True), ("ppo", False), ("gspo", False)],
)
@pytest.mark.parametrize("normalizer", ["local", "python", "tensor"])
def test_empty_normalizer_training_step_has_zero_loss_and_gradients(kind, token_level, normalizer):
    with torch.random.fork_rng():
        torch.manual_seed(17)
        model = torch.nn.Linear(3, 5)
        inputs = torch.randn(2, 4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = [p.detach().clone() for p in model.parameters()]
    output = model(inputs)
    mask = torch.zeros(output.shape[:2])
    info = {} if normalizer == "local" else get_loss_batch_info(object(), mask)
    if normalizer == "python":
        info = {key: value.item() if torch.is_tensor(value) else value for key, value in info.items()}

    if kind == "value":
        values = output[..., 0]
        loss = _loss_module.ValueLoss(clip_eps=0.2, token_level_loss=token_level)(
            values, values.detach(), torch.ones_like(values), mask, **info
        )
    else:
        logps = output.log_softmax(-1)[..., 0]
        if kind == "sft":
            loss = _loss_module.SFTLoss(token_level_loss=token_level)(logps, mask, **info)
        else:
            loss, clip_ratio, ppo_kl, _, _ = PolicyLoss(policy_loss_type=kind, token_level_loss=token_level)(
                logps, logps.detach() + 0.1, torch.ones_like(logps), mask, **info
            )
            torch.testing.assert_close(clip_ratio, torch.zeros_like(clip_ratio))
            torch.testing.assert_close(ppo_kl, torch.zeros_like(ppo_kl))

    torch.testing.assert_close(loss, torch.zeros_like(loss))
    loss.backward()
    for parameter in model.parameters():
        torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter))
    optimizer.step()
    for parameter, expected in zip(model.parameters(), before):
        torch.testing.assert_close(parameter, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
@pytest.mark.parametrize("global_normalizer", [False, True])
def test_empty_normalizer_rows_do_not_poison_nonempty_sequences(dtype, global_normalizer):
    values = torch.tensor([[1.0, 3.0], [5.0, 7.0]], dtype=dtype, requires_grad=True)
    mask = torch.tensor([[1.0, 1.0], [0.0, 0.0]], dtype=dtype)
    info = get_loss_batch_info(object(), mask) if global_normalizer else {}
    loss = aggregate_loss(values, mask, token_level_loss=False, **info)
    torch.testing.assert_close(loss, torch.tensor(2.0, dtype=loss.dtype))
    loss.backward()
    torch.testing.assert_close(values.grad, torch.tensor([[0.5, 0.5], [0.0, 0.0]], dtype=dtype))


@pytest.mark.parametrize("token_level", [False, True])
@pytest.mark.parametrize("python_normalizer", [False, True])
def test_empty_normalizer_keeps_fractional_accumulation_normalizers(token_level, python_normalizer):
    parameter = torch.nn.Parameter(torch.tensor(0.75, dtype=torch.float64))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    masks = [torch.tensor([[1.0, 0.0]])] + [torch.zeros(1, 2) for _ in range(3)]
    losses = []
    for mask, info in iter_grad_accum_global_norm(iter(masks), object(), 4, lambda item: item):
        assert info["batch_num_tokens"].item() == 0.25
        assert info["global_batch_size"].item() == 0.25
        if python_normalizer:
            info = {key: value.item() if torch.is_tensor(value) else value for key, value in info.items()}
        loss = aggregate_loss(
            parameter * torch.tensor([[2.0, 3.0]], dtype=torch.float64), mask, token_level_loss=token_level, **info
        )
        losses.append(loss.detach() / 4)
        (loss / 4).backward()
    torch.testing.assert_close(sum(losses), torch.tensor(1.5, dtype=torch.float64))
    torch.testing.assert_close(parameter.grad, torch.tensor(2.0, dtype=torch.float64))
    optimizer.step()
    torch.testing.assert_close(parameter, torch.tensor(0.55, dtype=torch.float64))


@pytest.mark.parametrize("token_level", [False, True])
def test_empty_normalizer_local_shard_preserves_positive_global_normalization(token_level):
    full = torch.tensor([[1.0, 2.0], [3.0, 5.0]], dtype=torch.float64, requires_grad=True)
    sharded = full.detach().clone().requires_grad_()
    mask = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    info = get_loss_batch_info(object(), mask)
    expected = aggregate_loss(full, mask, token_level_loss=token_level, **info)
    info["dp_size"] = 2
    losses = [
        aggregate_loss(sharded[i : i + 1], mask[i : i + 1], token_level_loss=token_level, **info) for i in range(2)
    ]
    actual = sum(losses) / 2  # gradient averaging across two DP shards
    torch.testing.assert_close(losses[0], torch.zeros_like(losses[0]))
    torch.testing.assert_close(actual, expected)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(sharded.grad, full.grad)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("dim", [None, 0, -1])
def test_empty_normalizer_masked_mean_keeps_positive_fractional_weights(dtype, dim):
    for mask in (torch.zeros(2, 2, dtype=dtype), torch.tensor([[0.25, 0.0], [0.0, 0.0]], dtype=dtype)):
        values = torch.tensor([[2.0, 3.0], [4.0, 5.0]], dtype=dtype, requires_grad=True)
        result = _loss_module.masked_mean(values, mask, dim)
        expected = torch.zeros_like(result)
        if mask.sum() > 0:
            if dim is None:
                expected.fill_(2.0)
            else:
                expected[0] = 2.0
        torch.testing.assert_close(result, expected)
        result.sum().backward()
        torch.testing.assert_close(values.grad, (mask > 0).to(dtype))


@pytest.mark.parametrize("token_level", [False, True])
@pytest.mark.parametrize("global_normalizer", [False, True])
def test_empty_normalizer_does_not_hide_nan_at_valid_tokens(token_level, global_normalizer):
    values = torch.tensor([[float("nan"), 1.0]])
    mask = torch.ones_like(values)
    info = get_loss_batch_info(object(), mask) if global_normalizer else {}
    assert torch.isnan(aggregate_loss(values, mask, token_level_loss=token_level, **info))


def test_empty_normalizer_masked_mean_without_mask_keeps_native_mean():
    values = torch.tensor([[1.0, 3.0], [5.0, 7.0]], requires_grad=True)
    torch.testing.assert_close(_loss_module.masked_mean(values, None), values.mean())
