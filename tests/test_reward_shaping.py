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


@pytest.mark.parametrize("gamma", [0.0, 0.5, 0.95, 1.0])
def test_reinforce_discount_skips_non_action_observation_tokens(maker_module, gamma):
    maker = maker_module.RemoteExperienceMaker.__new__(maker_module.RemoteExperienceMaker)
    rewards = torch.tensor([[0.0, 0.0, 1.0]])
    action_mask = torch.tensor([[True, False, True]])

    returns = maker.get_cumulative_returns(rewards, action_mask, gamma)

    expected = torch.tensor([[gamma, 0.0, 1.0]])
    torch.testing.assert_close(returns * action_mask, expected)


def test_reinforce_mask_none_keeps_position_discount(maker_module):
    maker = maker_module.RemoteExperienceMaker.__new__(maker_module.RemoteExperienceMaker)
    rewards = torch.tensor([[0.2, 0.0, 1.0]])

    returns = maker.get_cumulative_returns(rewards, None, 0.5)

    torch.testing.assert_close(returns, torch.tensor([[0.45, 0.5, 1.0]]))


def test_reinforce_action_returns_handle_mixed_gaps_padding_and_kl_rewards(maker_module):
    maker = maker_module.RemoteExperienceMaker.__new__(maker_module.RemoteExperienceMaker)
    rewards = torch.tensor(
        [
            [0.2, 9.0, 1.0, 8.0, 8.0, 8.0],
            [0.3, 0.4, 7.0, 7.0, 7.0, 7.0],
            [0.1, 6.0, 6.0, 0.5, 6.0, 1.5],
        ]
    )
    action_mask = torch.tensor(
        [
            [True, False, True, False, False, False],
            [True, True, False, False, False, False],
            [True, False, False, True, False, True],
        ]
    )

    returns = maker.get_cumulative_returns(rewards, action_mask, 0.5)

    expected = torch.tensor(
        [
            [0.7, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.5, 0.4, 0.0, 0.0, 0.0, 0.0],
            [0.725, 0.0, 0.0, 1.25, 0.0, 1.5],
        ]
    )
    torch.testing.assert_close(returns * action_mask, expected)


def test_reinforce_action_returns_reach_policy_loss_gradient(maker_module):
    maker = maker_module.RemoteExperienceMaker.__new__(maker_module.RemoteExperienceMaker)
    rewards = torch.tensor([[0.0, 0.0, 1.0]])
    action_mask = torch.tensor([[True, False, True]])
    returns = maker.get_cumulative_returns(rewards, action_mask, 0.5)
    log_probs = torch.zeros_like(returns, requires_grad=True)

    loss, *_ = _loss_module.PolicyLoss()(log_probs, log_probs.detach(), returns, action_mask)
    loss.backward()

    torch.testing.assert_close(log_probs.grad, torch.tensor([[-0.25, 0.0, -0.5]]))


def test_reinforce_processing_preserves_masked_kl_and_policy_gradient(maker_module):
    maker = maker_module.RemoteExperienceMaker.__new__(maker_module.RemoteExperienceMaker)
    maker.advantage_estimator = "reinforce"
    maker.kl_ctl = SimpleNamespace(value=1.0)
    maker.strategy = SimpleNamespace(
        args=SimpleNamespace(
            rollout=SimpleNamespace(n_samples_per_prompt=1),
            reward=SimpleNamespace(clip_range=None),
            algo=SimpleNamespace(advantage=SimpleNamespace(estimator="reinforce", gamma=0.5, no_std_norm=True)),
        )
    )
    experience = Experience(
        index=[0],
        rewards=torch.tensor([1.0]),
        action_mask=torch.tensor([[True, False, True]]),
        kl=torch.tensor([[0.1, 0.0, 0.2]]),
        values=torch.zeros(1, 3),
        info={"reward": torch.tensor([1.0])},
    )

    result = maker.compute_advantages_and_returns([experience])[0]
    log_probs = torch.zeros_like(result.advantages, requires_grad=True)
    loss, *_ = _loss_module.PolicyLoss()(log_probs, log_probs.detach(), result.advantages, result.action_mask)
    loss.backward()

    torch.testing.assert_close(result.returns * result.action_mask, torch.tensor([[0.3, 0.0, 0.8]]))
    torch.testing.assert_close(log_probs.grad, torch.tensor([[0.125, 0.0, -0.125]]))
