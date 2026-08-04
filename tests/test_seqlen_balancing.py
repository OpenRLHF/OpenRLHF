import importlib.util
from pathlib import Path

import pytest


def _load_seqlen_balancing_module():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "openrlhf.utils.seqlen_balancing", root / "openrlhf" / "utils" / "seqlen_balancing.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


get_minimum_num_micro_batch_size = _load_seqlen_balancing_module().get_minimum_num_micro_batch_size


def test_micro_batch_sizing_rejects_sequence_over_effective_capacity():
    with pytest.raises(ValueError, match=r"Sequence length 21 exceeds .* capacity 20"):
        get_minimum_num_micro_batch_size(
            total_lengths=[21],
            max_tokens_per_gpu=10,
            ring_attn_size=2,
            ds_tensor_parallel_size=1,
        )


def test_micro_batch_sizing_accepts_sequence_at_effective_capacity():
    num_batches = get_minimum_num_micro_batch_size(
        total_lengths=[20],
        max_tokens_per_gpu=10,
        ring_attn_size=2,
        ds_tensor_parallel_size=1,
    )

    assert num_batches == 1
