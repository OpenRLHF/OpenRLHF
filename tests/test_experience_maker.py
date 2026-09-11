from types import SimpleNamespace

import torch

from openrlhf.trainer.ppo_utils.experience import Experience
from openrlhf.trainer.ppo_utils.experience_maker import RemoteExperienceMaker


def _make_args(estimator="dr_grpo", n_samples_per_prompt=2):
    return SimpleNamespace(
        rollout=SimpleNamespace(n_samples_per_prompt=n_samples_per_prompt),
        reward=SimpleNamespace(clip_range=None, overlong_buffer_len=None, stop_properly_penalty_coef=None),
        algo=SimpleNamespace(
            advantage=SimpleNamespace(estimator=estimator, gamma=1.0, lambd=1.0, no_std_norm=True),
        ),
    )


def _make_maker(estimator="dr_grpo", n_samples_per_prompt=2):
    args = _make_args(estimator=estimator, n_samples_per_prompt=n_samples_per_prompt)
    maker = object.__new__(RemoteExperienceMaker)
    maker.strategy = SimpleNamespace(args=args)
    maker.args = args
    maker.advantage_estimator = args.algo.advantage.estimator
    maker.kl_ctl = SimpleNamespace(value=0.0)
    return maker


def _experience(index, reward, action_mask):
    action_mask = torch.tensor([action_mask], dtype=torch.bool)
    return Experience(
        action_mask=action_mask,
        rewards=torch.tensor([reward], dtype=torch.float32),
        kl=torch.zeros_like(action_mask, dtype=torch.float32),
        values=torch.zeros_like(action_mask, dtype=torch.float32),
        index=[index],
        info={},
    )


def test_dr_grpo_group_baseline_ignores_zero_action_rewards():
    zero_action = _experience(0, 100.0, [0, 0, 0])
    valid_action = _experience(1, 1.0, [1, 1, 1])

    _, valid_after = _make_maker("dr_grpo").compute_advantages_and_returns([zero_action, valid_action])

    assert torch.allclose(valid_after.advantages, torch.zeros_like(valid_after.advantages))
    assert torch.allclose(valid_after.info["return"], torch.tensor([0.0]))


def test_dr_grpo_group_baseline_uses_valid_siblings_only():
    zero_action = _experience(0, 100.0, [0, 0])
    valid_low = _experience(1, 1.0, [1, 1])
    valid_high = _experience(2, 3.0, [1, 1])

    _, low_after, high_after = _make_maker("dr_grpo", n_samples_per_prompt=3).compute_advantages_and_returns(
        [zero_action, valid_low, valid_high],
    )

    assert torch.allclose(low_after.advantages, torch.tensor([[-1.0, -1.0]]))
    assert torch.allclose(high_after.advantages, torch.tensor([[1.0, 1.0]]))
