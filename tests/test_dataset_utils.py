import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


class _FakeDataset:
    def __len__(self):
        return 1

    def __contains__(self, key):
        return False

    def select(self, indices):
        return self


def _load_dataset_utils_module(monkeypatch, calls):
    datasets_module = types.ModuleType("datasets")

    def load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return _FakeDataset()

    datasets_module.load_dataset = load_dataset
    datasets_module.load_from_disk = lambda path: _FakeDataset()
    datasets_module.interleave_datasets = lambda data, **kwargs: data[0]
    datasets_module.concatenate_datasets = lambda data: data[0]
    monkeypatch.setitem(sys.modules, "datasets", datasets_module)

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "openrlhf.datasets.utils", root / "openrlhf" / "datasets" / "utils.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("suffix", "loader_name"),
    [
        ("JSON", "json"),
        ("JSONL", "json"),
        ("CSV", "csv"),
        ("PARQUET", "parquet"),
        ("ARROW", "arrow"),
    ],
)
def test_blending_datasets_normalizes_local_file_extension(monkeypatch, suffix, loader_name):
    calls = []
    module = _load_dataset_utils_module(monkeypatch, calls)
    strategy = SimpleNamespace(
        args=SimpleNamespace(use_ms=False),
        print=lambda message: None,
        is_rank_0=lambda: False,
    )
    path = f"/tmp/train.{suffix}"

    module.blending_datasets(path, strategy=strategy)

    assert calls[0] == ((loader_name,), {"data_files": path})
