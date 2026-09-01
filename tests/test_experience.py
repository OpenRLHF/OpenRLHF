import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F


def _load_experience_module():
    root = Path(__file__).resolve().parents[1]

    utils_module = types.ModuleType("openrlhf.utils.utils")

    def zero_pad_sequences(sequences, side="left", value=0, stack=False):
        max_len = max(sequence.size(-1) for sequence in sequences)
        padded = []
        for sequence in sequences:
            pad_len = max_len - sequence.size(-1)
            padding = (pad_len, 0) if side == "left" else (0, pad_len)
            padded.append(F.pad(sequence, padding, value=value))
        return torch.stack(padded) if stack else torch.cat(padded)

    utils_module.zero_pad_sequences = zero_pad_sequences
    original_utils_module = sys.modules.get("openrlhf.utils.utils")
    sys.modules["openrlhf.utils.utils"] = utils_module
    try:
        spec = importlib.util.spec_from_file_location(
            "_openrlhf_experience_test", root / "openrlhf" / "trainer" / "ppo_utils" / "experience.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if original_utils_module is None:
            sys.modules.pop("openrlhf.utils.utils", None)
        else:
            sys.modules["openrlhf.utils.utils"] = original_utils_module


_experience_module = _load_experience_module()
Experience = _experience_module.Experience
balance_experiences = _experience_module.balance_experiences


def _make_experience(index, length):
    return Experience(
        sequences=torch.full((1, length), index, dtype=torch.long),
        attention_mask=torch.ones((1, length), dtype=torch.long),
        action_mask=torch.ones((1, length - 1), dtype=torch.bool),
        total_length=torch.tensor([length]),
        prompts=[f"sample-{index}"],
    )


@pytest.mark.parametrize("sample_count", [4, 5, 9])
def test_balance_experiences_preserves_divisible_and_ragged_batches(sample_count):
    args = SimpleNamespace(
        actor=SimpleNamespace(num_nodes=1, num_gpus_per_node=4),
        ds=SimpleNamespace(ring_attn_size=1, tensor_parallel_size=1),
    )
    inputs = [_make_experience(index, sample_count - index + 1) for index in range(sample_count)]

    outputs = balance_experiences(inputs, args)

    output_prompts = [prompt for batch in outputs for prompt in batch.prompts]
    batch_sizes = [len(batch.sequences) for batch in outputs]
    assert len(outputs) == 4
    assert all(size > 0 for size in batch_sizes)
    assert sum(batch_sizes) == sample_count
    assert sorted(output_prompts) == sorted(f"sample-{index}" for index in range(sample_count))


def _load_ray_batch_modules():
    root = Path(__file__).resolve().parents[1]
    fake_ray = MagicMock()
    fake_ray.remote.side_effect = lambda obj=None, **_: (lambda cls: cls) if obj is None else obj

    placement_group_module = types.ModuleType("ray.util.placement_group")
    placement_group_module.PlacementGroup = object
    placement_group_module.placement_group = MagicMock()
    scheduling_module = types.ModuleType("ray.util.scheduling_strategies")
    scheduling_module.PlacementGroupSchedulingStrategy = MagicMock
    experience_module = types.ModuleType("openrlhf.trainer.ppo_utils.experience")
    experience_module.Experience = Experience

    stubs = {
        "ray": fake_ray,
        "ray.util": types.ModuleType("ray.util"),
        "ray.util.placement_group": placement_group_module,
        "ray.util.scheduling_strategies": scheduling_module,
        "openrlhf.models": MagicMock(),
        "openrlhf.models.utils": MagicMock(),
        "openrlhf.trainer.ppo_utils.experience": experience_module,
        "openrlhf.trainer.ppo_utils.length_penalty": MagicMock(),
        "openrlhf.trainer.ray.launcher": None,
        "openrlhf.trainer.ray.utils": MagicMock(),
        "openrlhf.utils.deepspeed": MagicMock(),
        "openrlhf.utils.logging_utils": MagicMock(),
        "openrlhf.utils.seqlen_balancing": MagicMock(),
    }
    originals = {name: sys.modules.get(name) for name in stubs}
    try:
        for name, module in stubs.items():
            if module is not None:
                sys.modules[name] = module

        launcher_spec = importlib.util.spec_from_file_location(
            "_openrlhf_launcher_test", root / "openrlhf" / "trainer" / "ray" / "launcher.py"
        )
        launcher_module = importlib.util.module_from_spec(launcher_spec)
        sys.modules[launcher_spec.name] = launcher_module
        launcher_spec.loader.exec_module(launcher_module)
        sys.modules["openrlhf.trainer.ray.launcher"] = launcher_module

        maker_spec = importlib.util.spec_from_file_location(
            "_openrlhf_experience_maker_test",
            root / "openrlhf" / "trainer" / "ppo_utils" / "experience_maker.py",
        )
        maker_module = importlib.util.module_from_spec(maker_spec)
        sys.modules[maker_spec.name] = maker_module
        maker_spec.loader.exec_module(maker_module)
        return maker_module, launcher_module, fake_ray
    finally:
        for name, module in originals.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


_experience_maker_module, _launcher_module, _fake_ray = _load_ray_batch_modules()


@pytest.mark.parametrize(
    ("sample_count", "group_counts", "expected_batch_sizes", "expected_distribution"),
    [
        (672, [4, 4, 4, 4], ([16] * 10 + [8]) * 4, {4: (11, 168)}),
        (720, [4, 6], ([16] * 3 + [12]) * 12, {4: (12, 180), 6: (8, 120)}),
    ],
)
def test_split_rollout_samples_balances_every_effective_actor_group(
    sample_count, group_counts, expected_batch_sizes, expected_distribution
):
    groups = [SimpleNamespace(_actor_handlers=[object()] * (count * 2), duplicate_actors=2) for count in group_counts]
    maker = _experience_maker_module.RemoteExperienceMaker.__new__(_experience_maker_module.RemoteExperienceMaker)
    maker.actor_model_group = groups[0]
    maker.critic_model_group = groups[1] if len(groups) > 1 else None
    maker.initial_model_group = groups[2] if len(groups) > 2 else None
    maker.reward_model_group = groups[3] if len(groups) > 3 else None
    maker.args = SimpleNamespace(
        train=SimpleNamespace(dynamic_batch_enable=False), rollout=SimpleNamespace(micro_batch_size=16)
    )
    maker.tokenizer = SimpleNamespace(pad_token_id=0)
    samples = [Experience(prompts=[f"sample-{index}"]) for index in range(sample_count)]

    batches = maker.split_rollout_samples(samples)

    assert [len(batch.index) for batch in batches] == expected_batch_sizes
    indexes = [index for batch in batches for index in batch.index]
    assert indexes == list(range(sample_count))
    assert len(set(indexes)) == sample_count
    for actor_count, (expected_calls, expected_samples) in expected_distribution.items():
        calls_per_actor = len(batches) // actor_count
        samples_per_actor = [
            sum(len(batch.index) for batch in batches[start : start + calls_per_actor])
            for start in range(0, len(batches), calls_per_actor)
        ]
        assert calls_per_actor == expected_calls
        assert samples_per_actor == [expected_samples] * actor_count


def test_split_rollout_samples_rejects_nondivisible_group_lcm():
    group = SimpleNamespace(_actor_handlers=[object()] * 4, duplicate_actors=1)
    maker = _experience_maker_module.RemoteExperienceMaker.__new__(_experience_maker_module.RemoteExperienceMaker)
    maker.actor_model_group = group
    maker.critic_model_group = maker.initial_model_group = maker.reward_model_group = None
    maker.args = SimpleNamespace(
        train=SimpleNamespace(dynamic_batch_enable=False), rollout=SimpleNamespace(micro_batch_size=4)
    )
    maker.tokenizer = SimpleNamespace(pad_token_id=0)

    with pytest.raises(ValueError, match="sample_count=10, group_lcm=4"):
        maker.split_rollout_samples([Experience() for _ in range(10)])


def test_async_run_method_batch_rejects_uneven_work_items():
    group = _launcher_module.RayActorGroup.__new__(_launcher_module.RayActorGroup)
    group._actor_handlers = [object()] * 4
    group.duplicate_actors = 1
    _fake_ray.put.reset_mock()

    with pytest.raises(ValueError, match="total_length=42, effective_actors=4"):
        group.async_run_method_batch("forward", sequences=list(range(42)))

    _fake_ray.put.assert_not_called()
