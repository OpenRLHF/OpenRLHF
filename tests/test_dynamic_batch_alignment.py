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


align_num_micro_batches = _load_seqlen_balancing_module().align_num_micro_batches


def test_micro_batch_alignment_rounds_minimum_up_to_actor_multiple():
    assert align_num_micro_batches(minimum_batches=5, effective_actors=4, num_samples=8) == 8


def test_micro_batch_alignment_keeps_feasible_actor_minimum():
    assert align_num_micro_batches(minimum_batches=3, effective_actors=4, num_samples=4) == 4


def test_micro_batch_alignment_rejects_empty_required_partitions():
    with pytest.raises(ValueError, match="Cannot create 8 non-empty micro-batches from 5 samples"):
        align_num_micro_batches(minimum_batches=5, effective_actors=4, num_samples=5)
