import importlib.util
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "_openrlhf_seqlen_balancing_test",
    Path(__file__).resolve().parents[1] / "openrlhf" / "utils" / "seqlen_balancing.py",
)
seqlen_balancing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(seqlen_balancing)


@pytest.mark.parametrize("ring_size,tp_size", [(1, 1), (2, 1), (1, 2), (2, 2)])
def test_balancing_preserves_the_first_fit_token_limit(ring_size, tp_size):
    scale = 1024 * ring_size * tp_size
    lengths = [length * scale for length in [6, 7, 9, 7, 8]]
    limit = 20 * scale
    count = seqlen_balancing.get_minimum_num_micro_batch_size(lengths, 20 * 1024, ring_size, tp_size)
    assert count == 2

    partitions = seqlen_balancing.get_seqlen_balanced_partitions(lengths, count, False, max_partition_sum=limit)

    assert len(partitions) == count
    assert sorted(idx for partition in partitions for idx in partition) == list(range(len(lengths)))
    assert all(sum(lengths[idx] for idx in partition) <= limit for partition in partitions)


def test_token_limit_preserves_a_larger_synchronized_partition_count():
    rng = random.Random(42)
    for _ in range(100):
        lengths = [rng.randint(1, 100) for _ in range(16)]
        minimum = seqlen_balancing.get_minimum_num_micro_batch_size(lengths, 100, 1, 1)
        for count in range(minimum, len(lengths) + 1):
            partitions = seqlen_balancing.get_seqlen_balanced_partitions(lengths, count, False, max_partition_sum=100)
            assert len(partitions) == count
            assert all(partition for partition in partitions)
            assert sorted(idx for partition in partitions for idx in partition) == list(range(len(lengths)))
            assert all(sum(lengths[idx] for idx in partition) <= 100 for partition in partitions)


def test_token_limit_keeps_balanced_partitions_when_they_fit():
    lengths = [1, 3, 5, 2, 4]
    expected = seqlen_balancing.get_seqlen_balanced_partitions(lengths, 2, False)
    actual = seqlen_balancing.get_seqlen_balanced_partitions(lengths, 2, False, max_partition_sum=10)
    assert actual == expected


def test_equal_size_balancing_remains_unchanged():
    lengths = [1, 3, 5, 2, 4, 6]
    partitions = seqlen_balancing.get_seqlen_balanced_partitions(lengths, 2, True)
    assert [len(partition) for partition in partitions] == [3, 3]


@pytest.mark.parametrize("lengths,count", [([101, 1], 2), ([80, 80, 80], 2)])
def test_unachievable_token_limit_is_reported(lengths, count):
    with pytest.raises(ValueError):
        seqlen_balancing.get_seqlen_balanced_partitions(lengths, count, False, max_partition_sum=100)


def test_replay_buffer_training_partitions_obey_the_token_limit(monkeypatch):
    import torch
    from torch import distributed as dist

    from openrlhf.trainer.ppo_utils.experience import Experience
    from openrlhf.trainer.ppo_utils.replay_buffer import NaiveReplayBuffer

    lengths = [6 * 1024, 7 * 1024, 9 * 1024, 7 * 1024, 8 * 1024]
    args = SimpleNamespace(
        train=SimpleNamespace(batch_size=5, max_tokens_per_gpu=20 * 1024),
        rollout=SimpleNamespace(batch_size=5, n_samples_per_prompt=1),
        ds=SimpleNamespace(ring_attn_size=1, tensor_parallel_size=1),
    )
    # Exercise production setup on CPU, replacing only device selection and the
    # single-rank communication boundary.
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(dist, "get_world_size", lambda **kwargs: 1)
    monkeypatch.setattr(dist, "all_reduce", lambda *args, **kwargs: None)
    strategy = SimpleNamespace(
        args=args,
        ds_device_mesh={"dp": SimpleNamespace(get_group=lambda: None)},
        all_reduce=lambda tensor, op: tensor,
    )
    buffer = NaiveReplayBuffer.__new__(NaiveReplayBuffer)
    buffer.items = [
        Experience(total_length=torch.tensor([length]), action_mask=torch.ones((1, 2), dtype=torch.bool))
        for length in lengths
    ]
    buffer.setup_dynamic_batch(strategy)

    assert len(buffer.dynamic_indices) == 2
    assert sorted(idx for partition in buffer.dynamic_indices for idx in partition) == list(range(5))
    assert max(sum(lengths[idx] for idx in partition) for partition in buffer.dynamic_indices) <= 20 * 1024
    assert buffer.dynamic_optimizer_step == [0, 1]
    assert sum(buffer.dynamic_sample_loss_scale) == pytest.approx(1.0)
    assert buffer.dynamic_global_batch_size == [5.0, 5.0]
    assert buffer.dynamic_batch_num_tokens == [10.0, 10.0]
