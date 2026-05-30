from types import SimpleNamespace

import pytest
import torch

from openrlhf.trainer.ppo_utils.length_penalty import apply_overlong_penalty


def _experience(reward, response_length=None, action_mask=None):
    return SimpleNamespace(
        rewards=torch.tensor([reward], dtype=torch.float),
        response_length=torch.tensor([response_length]) if response_length is not None else None,
        action_mask=action_mask,
    )


def test_overlong_penalty_uses_action_mask_to_exclude_tool_response_tokens():
    experience = _experience(
        reward=1.0,
        response_length=10,
        action_mask=torch.tensor([[1, 1, 0, 0, 0, 1, 1, 1]], dtype=torch.bool),
    )

    num_penalized = apply_overlong_penalty([experience], max_new_tokens=8, overlong_buffer_len=3)

    assert num_penalized == 0
    assert torch.allclose(experience.rewards, torch.tensor([1.0]))


def test_overlong_penalty_applies_to_trainable_action_tokens():
    experience = _experience(
        reward=1.0,
        response_length=10,
        action_mask=torch.tensor([[1, 1, 1, 0, 0, 1, 1, 1, 1]], dtype=torch.bool),
    )

    num_penalized = apply_overlong_penalty([experience], max_new_tokens=8, overlong_buffer_len=3)

    assert num_penalized == 1
    assert torch.allclose(experience.rewards, torch.tensor([1.0 - 2.0 / 3.0]))


def test_overlong_penalty_falls_back_to_response_length_without_action_mask():
    experience = _experience(reward=1.0, response_length=7, action_mask=None)

    num_penalized = apply_overlong_penalty([experience], max_new_tokens=8, overlong_buffer_len=3)

    assert num_penalized == 1
    assert torch.allclose(experience.rewards, torch.tensor([1.0 - 2.0 / 3.0]))


def test_overlong_penalty_requires_length_information():
    experience = _experience(reward=1.0)

    with pytest.raises(ValueError, match="action_mask.*response_length"):
        apply_overlong_penalty([experience], max_new_tokens=8, overlong_buffer_len=3)
