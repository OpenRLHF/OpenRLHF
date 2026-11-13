from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from openrlhf.trainer.ppo_utils import (
    Experience,
    apply_length_penalties,
    apply_overlong_penalty,
    apply_stop_properly_penalty,
)

_DTYPES = [torch.float16, torch.bfloat16, torch.float32, torch.float64]


def _strided(values, dtype):
    backing = torch.full((2 * len(values),), 17.0, dtype=dtype)
    backing[::2] = torch.tensor(values, dtype=dtype)
    return backing, backing[::2]


def _assert_equal(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(torch.signbit(actual[actual == 0]), torch.signbit(expected[expected == 0]))


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize(
    "buffer,factor,penalties,count",
    [
        (4, 1.0, [0, 0, -0.25, -0.5, -0.75, -1, -1], 5),
        (2.5, 0.3, [0, 0, 0, -0.2 * 0.3, -0.6 * 0.3, -0.3, -0.3], 4),
        (0, 1.0, [0] * 7, 0),
        (-1, 1.0, [0] * 7, 0),
        (4, 0.0, [0] * 7, 5),
        (4, -0.7, [0, 0, 0.25 * 0.7, 0.5 * 0.7, 0.75 * 0.7, 0.7, 0.7], 5),
    ],
)
def test_overlong_penalties_match_piecewise_reference(dtype, buffer, factor, penalties, count):
    backing, rewards = _strided([-0.0, -2.0, 0.25, -0.5, 1.0, 2.5, -4.0], dtype)
    lengths_backing, lengths = _strided([0, 8, 9, 10, 11, 12, 15], torch.float64)
    original_lengths = lengths_backing.clone()
    expected = rewards.clone()
    if count:
        first = 7 - count
        expected[first:] += torch.tensor(penalties[first:], dtype=dtype)
    experience = Experience(rewards=rewards, response_length=lengths)
    pointer = rewards.data_ptr()

    assert apply_overlong_penalty([experience], 12, buffer, factor) == count
    assert experience.rewards is rewards and rewards.data_ptr() == pointer
    _assert_equal(rewards, expected)
    torch.testing.assert_close(backing[1::2], torch.full((7,), 17.0, dtype=dtype), rtol=0, atol=0)
    torch.testing.assert_close(lengths_backing, original_lengths, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("coefficient", [-0.5, 0.0, 0.25, 1.0])
def test_stop_penalty_preserves_unselected_values_and_views(dtype, coefficient):
    values = [-0.0, float("nan"), float("inf"), 2.0, -2.0, float("-inf"), -3.0]
    flags = torch.tensor([False, True, False, True, False, True, True])
    backing, rewards = _strided(values, dtype)
    expected = rewards.clone()
    selected = [1, 3, 5, 6]
    expected[selected] = coefficient if coefficient < 0 else expected[selected] * coefficient
    experience = Experience(rewards=rewards, truncated=flags)
    pointer = rewards.data_ptr()

    assert apply_stop_properly_penalty([experience], coefficient) == 4
    assert experience.rewards is rewards and rewards.data_ptr() == pointer
    _assert_equal(rewards, expected)
    torch.testing.assert_close(backing[1::2], torch.full((7,), 17.0, dtype=dtype), rtol=0, atol=0)
    torch.testing.assert_close(flags, torch.tensor([False, True, False, True, False, True, True]))


def test_empty_and_missing_truncation_batches():
    empty = Experience(rewards=torch.empty(0), response_length=torch.empty(0), truncated=torch.empty(0))
    no_flags = Experience(rewards=torch.tensor([0.5]), truncated=None)
    assert apply_overlong_penalty([], 12, 4) == 0
    assert apply_overlong_penalty([empty], 12, 4) == 0
    assert apply_stop_properly_penalty([], 0.5) == 0
    assert apply_stop_properly_penalty([empty, no_flags], 0.5) == 0
    torch.testing.assert_close(no_flags.rewards, torch.tensor([0.5]))


def test_numeric_truncation_flags_and_integer_rewards_keep_assignment_semantics():
    experience = Experience(
        rewards=torch.tensor([3, 5, -5, 7], dtype=torch.int64),
        truncated=torch.tensor([0.0, 1.0, -2.0, float("nan")]),
    )
    assert apply_stop_properly_penalty([experience], 0.5) == 3
    torch.testing.assert_close(experience.rewards, torch.tensor([3, 2, -2, 3]))


def test_overlong_nan_lengths_and_inactive_integer_rewards():
    experience = Experience(
        rewards=torch.tensor([-0.0, float("nan"), 2.0, -1.0]),
        response_length=torch.tensor([float("nan"), 8.0, float("inf"), float("-inf")]),
    )
    assert apply_overlong_penalty([experience], 12, 4) == 1
    _assert_equal(experience.rewards, torch.tensor([-0.0, float("nan"), 1.0, -1.0]))
    integer = Experience(rewards=torch.tensor([1]), response_length=torch.tensor([8]))
    assert apply_overlong_penalty([integer], 12, 4) == 0
    integer.response_length.fill_(10)
    with pytest.raises(RuntimeError):
        apply_overlong_penalty([integer], 12, 4)


def test_invalid_configuration_still_fails_before_reward_changes():
    experience = Experience(
        rewards=torch.tensor([1.0]), response_length=torch.tensor([10]), truncated=torch.tensor([True])
    )
    with pytest.raises(AssertionError, match="max_new_tokens"):
        apply_overlong_penalty([experience], 4, 8)
    with pytest.raises(AssertionError, match="stop_properly_penalty_coef"):
        apply_stop_properly_penalty([experience], 1.5)
    torch.testing.assert_close(experience.rewards, torch.tensor([1.0]))


@pytest.mark.parametrize("kind", ["overlong", "stop"])
def test_length_penalties_extract_only_batch_scalar_counts(kind):
    experiences = [
        Experience(
            rewards=torch.linspace(-1, 1, 128),
            response_length=torch.arange(128) % 17,
            truncated=torch.arange(128) % 2 == 0,
        )
        for _ in range(2)
    ]
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        if kind == "overlong":
            apply_overlong_penalty(experiences, 12, 4)
        else:
            apply_stop_properly_penalty(experiences, 0.5)
    scalar_reads = sum(event.count for event in profile.key_averages() if event.key == "aten::_local_scalar_dense")
    assert scalar_reads <= len(experiences), scalar_reads


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("optimizer_type", [torch.optim.SGD, torch.optim.AdamW])
def test_combined_penalties_preserve_language_model_updates_and_resume(tmp_path, dtype, optimizer_type):
    torch.manual_seed(23)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=17,
            n_positions=8,
            n_embd=8,
            n_layer=1,
            n_head=2,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
    ).to(dtype)
    reference = deepcopy(model)
    kwargs = {"lr": 0.003, "momentum": 0.5} if optimizer_type is torch.optim.SGD else {"lr": 0.003}
    optimizer = optimizer_type(model.parameters(), **kwargs)
    reference_optimizer = optimizer_type(reference.parameters(), **kwargs)
    tokens = torch.tensor([[2, 3, 5, 7, 11], [3, 5, 7, 9, 13], [2, 4, 6, 8, 10], [3, 6, 9, 12, 15]])
    args = SimpleNamespace(
        rollout=SimpleNamespace(max_new_tokens=10),
        data=SimpleNamespace(max_len=20),
        reward=SimpleNamespace(overlong_buffer_len=4, overlong_penalty_factor=1.0, stop_properly_penalty_coef=0.5),
    )
    with torch.no_grad():
        old_log_probs = reference(tokens).logits[:, :-1].log_softmax(-1).gather(-1, tokens[:, 1:, None]).squeeze(-1)

    for step in range(3):
        rewards = torch.tensor([0.5, 0.25, -0.125, 1.0], dtype=dtype)
        experience = Experience(
            rewards=rewards,
            response_length=torch.tensor([4, 7, 9, 12]),
            truncated=torch.tensor([False, True, False, True]),
            info={"reward": rewards.clone()},
        )
        apply_length_penalties([experience], args)
        expected_rewards = torch.tensor([0.5, 0.0, -0.875, 0.0], dtype=dtype)
        torch.testing.assert_close(experience.rewards, expected_rewards, rtol=0, atol=0)
        torch.testing.assert_close(experience.info["reward"], expected_rewards, rtol=0, atol=0)
        assert experience.info["reward"].data_ptr() != experience.rewards.data_ptr()
        losses = []
        for current, opt, values in (
            (model, optimizer, experience.rewards),
            (reference, reference_optimizer, expected_rewards),
        ):
            opt.zero_grad()
            log_probs = current(tokens).logits[:, :-1].log_softmax(-1).gather(-1, tokens[:, 1:, None]).squeeze(-1)
            ratio = (log_probs - old_log_probs).exp()
            loss = -torch.minimum(ratio * values[:, None], ratio.clamp(0.8, 1.2) * values[:, None]).mean()
            loss.backward()
            losses.append(loss.detach())
        torch.testing.assert_close(losses[0], losses[1], rtol=0, atol=0)
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)
        optimizer.step()
        reference_optimizer.step()
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual_state, expected_state = optimizer.state_dict(), reference_optimizer.state_dict()
        assert actual_state["param_groups"] == expected_state["param_groups"]
        assert actual_state["state"].keys() == expected_state["state"].keys()
        for key, state in actual_state["state"].items():
            assert state.keys() == expected_state["state"][key].keys()
            for name, value in state.items():
                expected = expected_state["state"][key][name]
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(value, expected, rtol=0, atol=0)
                else:
                    assert value == expected

        if step == 0:
            path = tmp_path / "training.pt"
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, path)
            model = deepcopy(reference)
            optimizer = optimizer_type(model.parameters(), **kwargs)
            checkpoint = torch.load(path, weights_only=True)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
