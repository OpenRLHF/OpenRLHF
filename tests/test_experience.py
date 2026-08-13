import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

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
