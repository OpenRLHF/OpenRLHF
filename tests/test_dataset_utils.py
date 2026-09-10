import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from datasets import Dataset, DatasetDict


@pytest.mark.parametrize(
    "as_dict,split",
    [(False, "train"), (False, "validation"), (False, None), (True, "train"), (True, "validation")],
)
def test_saved_dataset_split_selection_does_not_scan_rows(tmp_path, monkeypatch, as_dict, split):
    spec = importlib.util.spec_from_file_location(
        "_dataset_utils_test", Path(__file__).resolve().parents[1] / "openrlhf/datasets/utils.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    train = Dataset.from_dict({"text": [f"train-{i}" for i in range(6)]})
    validation = Dataset.from_dict({"text": [f"validation-{i}" for i in range(3)]})
    data = DatasetDict(train=train, validation=validation) if as_dict else train
    data.save_to_disk(str(tmp_path / "data"))
    original_iter = Dataset.__iter__
    visited = []

    def track_rows(self):
        for row in original_iter(self):
            visited.append(row)
            yield row

    monkeypatch.setattr(Dataset, "__iter__", track_rows)
    strategy = SimpleNamespace(print=lambda *args: None, is_rank_0=lambda: False)
    result = module.blending_datasets(str(tmp_path / "data"), strategy=strategy, max_count=2, dataset_split=split)

    assert visited == []
    expected = validation if as_dict and split == "validation" else train
    assert result.to_dict() == expected.select(range(2)).to_dict()
