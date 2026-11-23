import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from datasets import Dataset, DatasetDict


@pytest.fixture
def dataset_utils():
    spec = importlib.util.spec_from_file_location(
        "_dataset_utils_test", Path(__file__).resolve().parents[1] / "openrlhf/datasets/utils.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "as_dict,split",
    [(False, "train"), (False, "validation"), (False, None), (True, "train"), (True, "validation")],
)
def test_saved_dataset_split_selection_does_not_scan_rows(tmp_path, monkeypatch, dataset_utils, as_dict, split):
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
    result = dataset_utils.blending_datasets(
        str(tmp_path / "data"), strategy=strategy, max_count=2, dataset_split=split
    )

    assert visited == []
    expected = validation if as_dict and split == "validation" else train
    assert result.to_dict() == expected.select(range(2)).to_dict()


def test_local_jsonl_loads_with_dpo_eval_split(tmp_path, dataset_utils):
    path = tmp_path / "eval.jsonl"
    path.write_text('{"chosen": "yes", "rejected": "no"}\n')
    strategy = SimpleNamespace(print=lambda *args: None, is_rank_0=lambda: False)

    result = dataset_utils.blending_datasets(str(path), strategy=strategy, dataset_split="test")

    assert result.to_dict() == {"chosen": ["yes"], "rejected": ["no"]}


def test_missing_named_split_reports_available_splits(tmp_path, dataset_utils):
    fold = Dataset.from_dict({"text": ["example"]})
    path = tmp_path / "dataset"
    DatasetDict(train=fold, validation=fold).save_to_disk(str(path))
    strategy = SimpleNamespace(print=lambda *args: None, is_rank_0=lambda: False)

    with pytest.raises(ValueError, match=r"does not contain split 'test'. Available splits: train, validation"):
        dataset_utils.blending_datasets(str(path), strategy=strategy, dataset_split="test")
