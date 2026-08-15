import importlib.util
from pathlib import Path

import pytest


def _load_batch_module():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "openrlhf.utils.deepspeed.batch", root / "openrlhf" / "utils" / "deepspeed" / "batch.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


calculate_gradient_accumulation_steps = _load_batch_module().calculate_gradient_accumulation_steps


def test_gradient_accumulation_accounts_for_data_parallel_size():
    assert calculate_gradient_accumulation_steps(128, 2, world_size=8) == 8
    assert calculate_gradient_accumulation_steps(8, 2, world_size=8, ring_attn_size=2) == 1


def test_dynamic_batch_uses_runtime_accumulation_control():
    assert calculate_gradient_accumulation_steps(8, 2, world_size=8, dynamic_batch=True) == 1


@pytest.mark.parametrize("train_batch_size", [8, 15])
def test_gradient_accumulation_rejects_zero_or_fractional_steps(train_batch_size):
    with pytest.raises(ValueError, match=r"micro_batch_size \(2\) \* data parallel size \(8\) = 16"):
        calculate_gradient_accumulation_steps(train_batch_size, 2, world_size=8)


def test_gradient_accumulation_rejects_incompatible_parallel_sizes():
    with pytest.raises(ValueError, match=r"world_size \(8\) must be divisible"):
        calculate_gradient_accumulation_steps(16, 2, world_size=8, ring_attn_size=3)
