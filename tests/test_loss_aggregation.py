import importlib.util
import sys
import types
from pathlib import Path

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


def test_aggregate_loss_ignores_masked_nonfinite_values():
    loss = torch.tensor([[1.0, float("nan")], [3.0, float("inf")]])
    mask = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

    assert torch.allclose(aggregate_loss(loss, mask), torch.tensor(2.0))
    assert torch.allclose(
        aggregate_loss(loss, mask, dp_size=1, batch_num_tokens=mask.sum()),
        torch.tensor(2.0),
    )
    assert torch.allclose(aggregate_loss(loss, mask, token_level_loss=False), torch.tensor(2.0))


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


def test_policy_loss_ignores_masked_nonfinite_old_log_probs():
    log_probs = torch.nn.Parameter(torch.tensor([[0.0, 0.0]], dtype=torch.float32))
    old_log_probs = torch.tensor([[0.0, float("nan")]], dtype=torch.float32)
    advantages = torch.ones_like(log_probs)
    mask = torch.tensor([[1.0, 0.0]], dtype=torch.float32)

    loss, clip_ratio, ppo_kl, vllm_kl = PolicyLoss()(log_probs, old_log_probs, advantages, action_mask=mask)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(clip_ratio)
    assert torch.isfinite(ppo_kl)
    assert vllm_kl is None
    assert torch.isfinite(log_probs.grad).all()
    assert torch.allclose(log_probs.grad, torch.tensor([[-1.0, 0.0]]))


def test_policy_loss_treats_missing_action_mask_as_all_actions():
    log_probs = torch.tensor([[-0.2, -0.4]], dtype=torch.float32)
    old_log_probs = torch.tensor([[-0.3, -0.1]], dtype=torch.float32)
    advantages = torch.tensor([[1.0, -0.5]], dtype=torch.float32)
    action_mask = torch.ones_like(log_probs)

    loss_fn = PolicyLoss(policy_loss_type="gspo")
    none_mask_outputs = loss_fn(log_probs, old_log_probs, advantages, action_mask=None)
    explicit_mask_outputs = loss_fn(log_probs, old_log_probs, advantages, action_mask=action_mask)

    for none_mask_value, explicit_mask_value in zip(none_mask_outputs[:3], explicit_mask_outputs[:3], strict=True):
        assert torch.allclose(none_mask_value, explicit_mask_value)


def test_policy_kl_metric_uses_policy_ratio_when_vllm_correction_is_enabled():
    log_probs = torch.tensor([[-0.2, -0.4]])
    old_log_probs = torch.tensor([[-0.3, -0.1]])
    rollout_log_probs = torch.tensor([[-1.3, -1.1]])
    advantages = torch.ones_like(log_probs)
    mask = torch.ones_like(log_probs)

    loss_fn = PolicyLoss(
        enable_vllm_is_correction=True,
        vllm_is_truncated_threshold=[0.1, 10.0],
        vllm_is_correction_type="tis",
    )
    _, _, ppo_kl, vllm_kl = loss_fn(
        log_probs,
        old_log_probs,
        advantages,
        action_mask=mask,
        rollout_log_probs=rollout_log_probs,
    )

    assert torch.allclose(ppo_kl, (old_log_probs - log_probs).mean())
    assert torch.allclose(vllm_kl, (rollout_log_probs - old_log_probs).mean())


def test_policy_kl_metric_is_not_clamped():
    log_probs = torch.tensor([[100.0]])
    old_log_probs = torch.zeros_like(log_probs)
    advantages = torch.ones_like(log_probs)
    mask = torch.ones_like(log_probs)

    _, _, ppo_kl, _ = PolicyLoss()(log_probs, old_log_probs, advantages, action_mask=mask)

    assert torch.allclose(ppo_kl, (old_log_probs - log_probs).mean())
