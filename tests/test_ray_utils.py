import importlib.util
from pathlib import Path

import pytest


def _load_ray_utils_module():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "openrlhf.trainer.ray.utils", root / "openrlhf" / "trainer" / "ray" / "utils.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


get_balanced_batch_ranges = _load_ray_utils_module().get_balanced_batch_ranges


def test_balanced_batch_ranges_keep_uneven_chunks_non_empty():
    ranges = get_balanced_batch_ranges(total_length=5, num_chunks=4)

    assert ranges == [(0, 2), (2, 3), (3, 4), (4, 5)]
    assert all(end > start for start, end in ranges)


@pytest.mark.parametrize(("total_length", "num_chunks"), [(0, 1), (3, 4), (4, 0)])
def test_balanced_batch_ranges_reject_invalid_sizes(total_length, num_chunks):
    with pytest.raises(ValueError):
        get_balanced_batch_ranges(total_length, num_chunks)
