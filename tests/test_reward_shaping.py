import copy
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from tests.test_agent import agent_modules  # noqa: F401
from tests.test_experience import Experience
from tests.test_loss_aggregation import _loss_module


@pytest.fixture
def maker_module(agent_modules, monkeypatch):
    monkeypatch.setitem(sys.modules, "openrlhf.models.utils", sys.modules[_loss_module.__package__ + ".utils"])
    monkeypatch.setitem(sys.modules, "openrlhf.trainer.ppo_utils.length_penalty", agent_modules[2])
    monkeypatch.setitem(sys.modules, "openrlhf.trainer.ray.launcher", MagicMock())
    monkeypatch.setitem(sys.modules, "openrlhf.utils.seqlen_balancing", MagicMock())
    path = Path(__file__).resolve().parents[1] / "openrlhf/trainer/ppo_utils/experience_maker.py"
    spec = importlib.util.spec_from_file_location("_reward_shaping_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "estimator", ["gae", "reinforce", "group_norm", "rloo", "reinforce_baseline", "dr_grpo", "flash_reinforce"]
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32, torch.int64])
def test_reward_shaping_matches_fp32_reference(maker_module, estimator, dtype):
    rewards = torch.tensor([1, 1, 1, 1.0078125] if dtype != torch.int64 else [0, 0, 0, 1], dtype=dtype)
    maker = maker_module.RemoteExperienceMaker.__new__(maker_module.RemoteExperienceMaker)
    maker.advantage_estimator = estimator
    maker.kl_ctl = SimpleNamespace(value=0.0)
    args = SimpleNamespace(
        rollout=SimpleNamespace(n_samples_per_prompt=4),
        reward=SimpleNamespace(clip_range=None),
        algo=SimpleNamespace(advantage=SimpleNamespace(estimator=estimator, gamma=1.0, lambd=1.0, no_std_norm=True)),
    )
    maker.strategy = SimpleNamespace(args=args)
    experience = Experience(
        index=list(range(4)),
        rewards=rewards,
        action_mask=torch.ones(4, 2),
        kl=torch.zeros(4, 2),
        values=torch.zeros(4, 2),
        info={"reward": rewards.clone()},
    )
    result = maker.compute_advantages_and_returns([experience])[0]
    reference = rewards.float() - rewards.float().mean()
    if estimator == "rloo":
        reference *= 4 / 3
    elif estimator == "group_norm":
        reference /= rewards.float().std() + 1e-9
    torch.testing.assert_close(result.advantages, reference[:, None].expand(-1, 2))
    assert result.rewards.dtype == result.info["reward"].dtype == torch.float32


@pytest.mark.parametrize("dtype", [torch.float16, torch.int64])
def test_constant_rewards_have_finite_zero_group_advantages(maker_module, dtype):
    maker = maker_module.RemoteExperienceMaker.__new__(maker_module.RemoteExperienceMaker)
    maker.advantage_estimator = "group_norm"
    maker.kl_ctl = SimpleNamespace(value=0.0)
    maker.strategy = SimpleNamespace(
        args=SimpleNamespace(
            rollout=SimpleNamespace(n_samples_per_prompt=2),
            reward=SimpleNamespace(clip_range=None),
            algo=SimpleNamespace(advantage=SimpleNamespace(estimator="group_norm", gamma=1.0)),
        )
    )
    experience = Experience(
        index=[0, 1], rewards=torch.zeros(2, dtype=dtype), action_mask=torch.ones(2, 1), kl=torch.zeros(2, 1), info={}
    )
    result = maker.compute_advantages_and_returns([experience])[0]
    torch.testing.assert_close(result.advantages, torch.zeros(2, 1))


@pytest.mark.parametrize("dtype,gap", [(torch.bfloat16, 1 / 128), (torch.float64, 1e-8)])
def test_group_advantages_produce_reference_policy_gradient(maker_module, dtype, gap):
    rewards = torch.tensor([1, 1, 1, 1 + gap], dtype=dtype)
    maker = maker_module.RemoteExperienceMaker.__new__(maker_module.RemoteExperienceMaker)
    maker.advantage_estimator = "group_norm"
    maker.kl_ctl = SimpleNamespace(value=0.0)
    maker.strategy = SimpleNamespace(
        args=SimpleNamespace(
            rollout=SimpleNamespace(n_samples_per_prompt=4),
            reward=SimpleNamespace(clip_range=None),
            algo=SimpleNamespace(advantage=SimpleNamespace(estimator="group_norm", gamma=1.0)),
        )
    )
    mask = torch.ones(4, 2)
    experience = Experience(index=list(range(4)), rewards=rewards, action_mask=mask, kl=torch.zeros(4, 2), info={})
    advantage = maker.compute_advantages_and_returns([experience])[0].advantages
    log_probs = torch.zeros(4, 2, requires_grad=True)
    loss, *_ = _loss_module.PolicyLoss()(log_probs, log_probs.detach(), advantage, mask)
    loss.backward()
    reference = (rewards.double() - rewards.double().mean()) / (rewards.double().std() + 1e-9)
    torch.testing.assert_close(advantage, reference.float()[:, None].expand(-1, 2))
    torch.testing.assert_close(log_probs.grad, -reference.float()[:, None].expand(-1, 2) / mask.sum())


def test_integer_rewards_preserve_fractional_length_penalty(maker_module):
    maker = maker_module.RemoteExperienceMaker.__new__(maker_module.RemoteExperienceMaker)
    maker.advantage_estimator = "dr_grpo"
    maker.kl_ctl = SimpleNamespace(value=0.0)
    maker.strategy = SimpleNamespace(
        args=SimpleNamespace(
            rollout=SimpleNamespace(n_samples_per_prompt=2, max_new_tokens=2),
            data=SimpleNamespace(max_len=3),
            reward=SimpleNamespace(clip_range=None, overlong_buffer_len=1, overlong_penalty_factor=0.5),
            algo=SimpleNamespace(advantage=SimpleNamespace(estimator="dr_grpo", gamma=1.0)),
        )
    )
    experience = Experience(
        index=[0, 1],
        rewards=torch.tensor([1, 1]),
        action_mask=torch.ones(2, 2),
        kl=torch.zeros(2, 2),
        response_length=torch.tensor([2, 1]),
        info={"reward": torch.tensor([1, 1])},
    )
    result = maker.compute_advantages_and_returns([experience])[0]
    torch.testing.assert_close(result.rewards, torch.tensor([0.5, 1.0]))
    torch.testing.assert_close(result.info["reward"], result.rewards)
    torch.testing.assert_close(result.advantages, torch.tensor([[-0.25, -0.25], [0.25, 0.25]]))


def test_flash_reinforce_centers_on_the_rollout_batch_mean_without_whitening(maker_module):
    """The single-rollout estimator: baseline = mean over ALL rollouts (n_samples_per_prompt=1 has no
    group) and no std normalization, so advantages == returns."""
    rewards = torch.tensor([1.0, 0.0, 0.0, 0.0])
    maker = maker_module.RemoteExperienceMaker.__new__(maker_module.RemoteExperienceMaker)
    maker.advantage_estimator = "flash_reinforce"
    maker.kl_ctl = SimpleNamespace(value=0.0)
    maker.strategy = SimpleNamespace(
        args=SimpleNamespace(
            rollout=SimpleNamespace(n_samples_per_prompt=1),
            reward=SimpleNamespace(clip_range=None),
            algo=SimpleNamespace(advantage=SimpleNamespace(estimator="flash_reinforce", gamma=1.0, no_std_norm=False)),
        )
    )
    experience = Experience(
        index=list(range(4)), rewards=rewards, action_mask=torch.ones(4, 2), kl=torch.zeros(4, 2), info={}
    )
    result = maker.compute_advantages_and_returns([experience])[0]
    torch.testing.assert_close(result.advantages, torch.tensor([0.75, -0.25, -0.25, -0.25])[:, None].expand(-1, 2))
    torch.testing.assert_close(result.advantages, result.returns)


@pytest.mark.parametrize("gamma", [0.0, 0.95, 1.0])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("mask_dtype", [None, torch.bool, torch.float32])
def test_cumulative_returns_match_discounted_suffix_sums(maker_module, gamma, dtype, mask_dtype):
    # Strided inputs cover padding, tool feedback gaps and a fully masked trajectory.
    rewards = ((torch.arange(4 * 34, dtype=dtype).reshape(4, 34) % 17 - 8) / 32)[:, ::2]
    assert not rewards.is_contiguous()
    rewards.requires_grad_()
    mask = torch.ones_like(rewards, dtype=torch.bool)
    mask[0, :3] = False
    mask[1, -4:] = False
    mask[2, 3:8] = False
    mask[3] = False
    mask = mask.to(mask_dtype) if mask_dtype is not None else None
    original_rewards = rewards.detach().clone()
    original_mask = mask.clone() if mask is not None else None

    actual = maker_module.RemoteExperienceMaker.get_cumulative_returns(None, rewards, mask, gamma)

    # Closed-form discounted suffix sums provide an independent FP64 oracle.
    masked = rewards.detach().double() * mask if mask is not None else rewards.detach().double()
    expected = torch.stack(
        [
            (masked[:, t:] * gamma ** torch.arange(rewards.size(1) - t, dtype=torch.float64)).sum(-1)
            for t in range(rewards.size(1))
        ],
        dim=1,
    ).to(dtype)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(rewards, original_rewards, rtol=0, atol=0)
    if mask is not None:
        torch.testing.assert_close(mask, original_mask, rtol=0, atol=0)
    assert actual.dtype == dtype
    assert actual.device == rewards.device
    assert not actual.requires_grad


@pytest.mark.parametrize("shape", [(0, 5), (3, 0), (2, 1)])
def test_undiscounted_returns_empty_and_single_token(maker_module, shape):
    rewards = torch.full(shape, 0.5)
    actual = maker_module.RemoteExperienceMaker.get_cumulative_returns(None, rewards, None, 1.0)
    torch.testing.assert_close(actual, rewards, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_undiscounted_long_returns_include_masked_gaps(maker_module, dtype):
    length = 65536
    mask = torch.ones(4, length, dtype=torch.bool)
    mask[0, :1024] = False
    mask[1, -1024:] = False
    mask[2, 8192:16384] = False
    mask[3] = False
    rewards = torch.full((4, length), -1 / 1024, dtype=dtype)
    terminal = torch.tensor([length - 1, length - 1025, length - 1, length - 1])
    rewards[torch.arange(4), terminal] += 1.0

    actual = maker_module.RemoteExperienceMaker.get_cumulative_returns(None, rewards, mask, 1.0)

    # Count future action tokens with a forward prefix count, including the current token.
    suffix_counts = mask.sum(-1, keepdim=True) - mask.long().cumsum(-1) + mask
    terminal_reward = (torch.arange(length)[None, :] <= terminal[:, None]).to(dtype)
    terminal_reward[3] = 0
    expected = terminal_reward - suffix_counts.to(dtype) / 1024
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "estimator", ["reinforce", "rloo", "reinforce_baseline", "group_norm", "dr_grpo", "flash_reinforce"]
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float64])
def test_cumulative_returns_preserve_advantages_and_policy_gradients(maker_module, monkeypatch, estimator, dtype):
    maker = maker_module.RemoteExperienceMaker.__new__(maker_module.RemoteExperienceMaker)
    maker.advantage_estimator = estimator
    maker.kl_ctl = SimpleNamespace(value=0.1)
    maker.strategy = SimpleNamespace(
        args=SimpleNamespace(
            rollout=SimpleNamespace(n_samples_per_prompt=1 if estimator == "flash_reinforce" else 2),
            reward=SimpleNamespace(clip_range=(-2, 2)),
            algo=SimpleNamespace(advantage=SimpleNamespace(estimator=estimator, gamma=1.0, no_std_norm=False)),
        )
    )
    mask = torch.tensor([[1, 1, 0, 0, 1, 1, 0], [1, 1, 1, 1, 1, 0, 0]], dtype=torch.bool)
    experiences = [
        Experience(
            index=indices,
            rewards=torch.tensor(rewards, dtype=dtype),
            action_mask=mask.clone(),
            kl=torch.linspace(
                -0.1, 0.2, mask.numel(), dtype=torch.float64 if dtype == torch.float64 else torch.float32
            ).reshape_as(mask),
            info={},
        )
        for indices, rewards in [([2, 0], [0.25, 1.0]), ([3, 1], [0.75, -0.5])]
    ]
    actual = maker.compute_advantages_and_returns(copy.deepcopy(experiences))

    def reference_returns(rewards, action_mask, gamma):
        masked = rewards.double() * action_mask
        return torch.stack([masked[:, t:].sum(-1) for t in range(masked.size(1))], dim=1).to(rewards.dtype)

    monkeypatch.setattr(maker, "get_cumulative_returns", reference_returns)
    expected = maker.compute_advantages_and_returns(experiences)
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result.returns, reference.returns)
        torch.testing.assert_close(result.advantages, reference.advantages)
        assert result.returns.dtype == (torch.float64 if dtype == torch.float64 else torch.float32)

    old_logps = torch.full((4, 7), -1.0)
    loss_fn = _loss_module.PolicyLoss()
    losses, gradients = [], []
    for results in (actual, expected):
        logits = torch.linspace(-0.5, 0.5, 4 * 7 * 3).reshape(4, 7, 3).requires_grad_()
        logps = logits.log_softmax(-1)[..., 0]
        advantages = torch.cat([result.advantages for result in results])
        loss, *_ = loss_fn(logps, old_logps, advantages, mask.repeat(2, 1))
        loss.backward()
        losses.append(loss.detach())
        gradients.append(logits.grad)
    torch.testing.assert_close(losses[0], losses[1])
    torch.testing.assert_close(gradients[0], gradients[1])
