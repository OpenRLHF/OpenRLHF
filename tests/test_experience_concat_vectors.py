from copy import deepcopy
from dataclasses import fields

import pytest
import torch

from openrlhf.trainer.ppo_utils.experience import Experience, make_experience_batch, split_experience_batch


def _experience(start, count, width):
    ids = torch.arange(start + 1, start + count + 1)
    sequences = ids[:, None].expand(-1, width).clone()
    actions = ids[:, None].expand(-1, width - 1).double() / 10
    return Experience(
        sequences=sequences,
        attention_mask=torch.ones_like(sequences),
        action_mask=torch.ones((count, width - 1), dtype=torch.bool),
        action_log_probs=-actions,
        base_action_log_probs=-actions - 0.1,
        rollout_log_probs=-actions - 0.2,
        values=actions,
        returns=actions + 0.3,
        advantages=actions + 0.4,
        kl=actions + 0.5,
        rewards=ids.double() / 10,
        scores=(ids % 2).float(),
        response_length=torch.full((count,), width - 1, dtype=torch.long),
        truncated=(ids % 2 == 0),
        total_length=torch.full((count,), width, dtype=torch.long),
        index=ids.tolist(),
        prompts=[f"prompt-{index}" for index in ids.tolist()],
        labels=[f"label-{index}" for index in ids.tolist()],
        info={"metric": ids.double() + 0.25},
    )


def _batches(sizes, different_widths):
    batches = []
    start = 0
    for index, count in enumerate(sizes):
        batches.append(_experience(start, count, 3 + index if different_widths else 4))
        start += count
    return batches


@pytest.mark.parametrize("sizes", [(2, 1), (1, 3), (2, 4, 1), (0, 2), (2, 2)])
@pytest.mark.parametrize("different_widths", [False, True])
def test_concat_preserves_episode_vector_alignment(sizes, different_widths):
    batches = _batches(sizes, different_widths)
    snapshots = deepcopy(batches)
    combined = Experience.concat_experiences(batches, pad_token_id=97)
    total = sum(sizes)
    assert len(combined.sequences) == total
    for field in fields(Experience):
        values = [getattr(batch, field.name) for batch in batches]
        actual = getattr(combined, field.name)
        if Experience.is_episode_tensor_field(field.name):
            expected = torch.cat(values)
            assert actual.shape == (total,)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        elif Experience.is_step_tensor_field(field.name):
            width = max(value.shape[-1] for value in values)
            expected = values[0].new_full((total, width), 97 if field.name == "sequences" else 0)
            offset = 0
            for value in values:
                expected[offset : offset + len(value), : value.shape[-1]] = value
                offset += len(value)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        combined.info["metric"], torch.cat([batch.info["metric"] for batch in batches]), rtol=0, atol=0
    )
    assert combined.prompts == [prompt for batch in batches for prompt in batch.prompts]
    assert combined.labels == [label for batch in batches for label in batch.labels]
    assert combined.index == [index for batch in batches for index in batch.index]
    for batch, original in zip(batches, snapshots):
        for field in fields(Experience):
            current, expected = getattr(batch, field.name), getattr(original, field.name)
            if isinstance(current, torch.Tensor):
                torch.testing.assert_close(current, expected, rtol=0, atol=0)
        torch.testing.assert_close(batch.info, original.info, rtol=0, atol=0)


@pytest.mark.parametrize("sizes", [(2, 1), (1, 3)])
@pytest.mark.parametrize("different_widths", [False, True])
def test_unequal_concatenated_batches_can_be_split_and_rebatched(sizes, different_widths):
    inputs = _batches(sizes, different_widths)
    combined = Experience.concat_experiences(inputs, pad_token_id=0)
    items = split_experience_batch(combined)
    assert len(items) == sum(sizes)
    expected_rewards = [reward for batch in inputs for reward in batch.rewards]
    for item, reward in zip(items, expected_rewards):
        torch.testing.assert_close(item.rewards, reward, rtol=0, atol=0)
    roundtrip = make_experience_batch(items)
    for field in fields(Experience):
        if Experience.is_episode_tensor_field(field.name) or Experience.is_step_tensor_field(field.name):
            torch.testing.assert_close(getattr(roundtrip, field.name), getattr(combined, field.name), rtol=0, atol=0)
    assert roundtrip.prompts == combined.prompts


@pytest.mark.parametrize("optimizer_cls", [torch.optim.SGD, torch.optim.Adam])
def test_value_training_uses_the_original_reward_for_every_concatenated_sample(optimizer_cls):
    combined = Experience.concat_experiences(_batches((2, 1), True), pad_token_id=0)
    features = (
        (combined.sequences.double() * combined.attention_mask).sum(dim=-1) / combined.total_length
    ).unsqueeze(-1)
    expected_features = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
    expected_rewards = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
    torch.manual_seed(17)
    model = torch.nn.Linear(1, 1, dtype=torch.float64)
    reference = deepcopy(model)
    kwargs = {"momentum": 0.9} if optimizer_cls is torch.optim.SGD else {}
    optimizer = optimizer_cls(model.parameters(), lr=0.02, **kwargs)
    expected_optimizer = optimizer_cls(reference.parameters(), lr=0.02, **kwargs)
    for _ in range(3):
        losses = []
        for current, current_optimizer, inputs, rewards in (
            (model, optimizer, features, combined.rewards),
            (reference, expected_optimizer, expected_features, expected_rewards),
        ):
            current_optimizer.zero_grad()
            predictions = current(inputs).squeeze(-1)
            assert predictions.shape == rewards.shape
            loss = (predictions - rewards).square().mean()
            loss.backward()
            losses.append(loss.detach())
            current_optimizer.step()
        torch.testing.assert_close(losses[0], losses[1], rtol=0, atol=0)
        for parameter, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(parameter.grad, expected.grad, rtol=0, atol=0)
        torch.testing.assert_close(model.state_dict(), reference.state_dict(), rtol=0, atol=0)
        torch.testing.assert_close(optimizer.state_dict(), expected_optimizer.state_dict(), rtol=0, atol=0)


def test_empty_experience_collection_still_returns_an_empty_experience():
    combined = Experience.concat_experiences([], pad_token_id=0)
    assert combined.sequences is None
    assert combined.rewards is None
    assert combined.info == {}
