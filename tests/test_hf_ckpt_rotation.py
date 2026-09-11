import importlib.util
import os
import time
from pathlib import Path

import pytest


def _load_ckpt_utils():
    root = Path(__file__).resolve().parents[1]
    path = root / "openrlhf" / "utils" / "ckpt_utils.py"
    spec = importlib.util.spec_from_file_location("_openrlhf_ckpt_utils", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ckpt_utils = _load_ckpt_utils()


def _make_export(ckpt_path: Path, name: str, mtime: float) -> Path:
    export_dir = ckpt_path / name
    export_dir.mkdir()
    (export_dir / "config.json").write_text("{}")
    os.utime(export_dir, (mtime, mtime))
    return export_dir


@pytest.fixture
def ckpt_path(tmp_path):
    return tmp_path


def _names(ckpt_path: Path):
    return sorted(p.name for p in ckpt_path.iterdir())


def test_evicts_oldest_regular_exports_beyond_max_num(ckpt_path):
    base = time.time() - 1000
    for i, step in enumerate((20, 40, 60)):
        _make_export(ckpt_path, f"global_step{step}_hf", base + i)

    removed = ckpt_utils.rotate_hf_checkpoints(str(ckpt_path), "global_step80", max_num=3)

    assert [os.path.basename(p) for p in removed] == ["global_step20_hf"]
    assert _names(ckpt_path) == ["global_step40_hf", "global_step60_hf"]


def test_keeps_all_when_under_max_num(ckpt_path):
    base = time.time() - 1000
    _make_export(ckpt_path, "global_step20_hf", base)

    removed = ckpt_utils.rotate_hf_checkpoints(str(ckpt_path), "global_step40", max_num=3)

    assert removed == []
    assert _names(ckpt_path) == ["global_step20_hf"]


def test_best_exports_do_not_count_toward_max_num(ckpt_path):
    base = time.time() - 1000
    _make_export(ckpt_path, "best_global_step10_hf", base)
    _make_export(ckpt_path, "global_step20_hf", base + 1)
    _make_export(ckpt_path, "global_step40_hf", base + 2)

    removed = ckpt_utils.rotate_hf_checkpoints(str(ckpt_path), "global_step60", max_num=3)

    assert removed == []
    assert _names(ckpt_path) == ["best_global_step10_hf", "global_step20_hf", "global_step40_hf"]


def test_new_best_evicts_old_best_only(ckpt_path):
    base = time.time() - 1000
    _make_export(ckpt_path, "best_global_step10_hf", base)
    _make_export(ckpt_path, "global_step20_hf", base + 1)
    _make_export(ckpt_path, "global_step40_hf", base + 2)

    removed = ckpt_utils.rotate_hf_checkpoints(str(ckpt_path), "best_global_step50", max_num=3)

    assert [os.path.basename(p) for p in removed] == ["best_global_step10_hf"]
    assert _names(ckpt_path) == ["global_step20_hf", "global_step40_hf"]


def test_ignores_non_hf_dirs_and_files(ckpt_path):
    base = time.time() - 1000
    (ckpt_path / "_actor").mkdir()
    (ckpt_path / "notes.txt").write_text("keep me")
    for i, step in enumerate((20, 40, 60)):
        _make_export(ckpt_path, f"global_step{step}_hf", base + i)

    removed = ckpt_utils.rotate_hf_checkpoints(str(ckpt_path), "global_step80", max_num=3)

    assert [os.path.basename(p) for p in removed] == ["global_step20_hf"]
    assert "_actor" in _names(ckpt_path)
    assert "notes.txt" in _names(ckpt_path)


def test_resaving_same_tag_is_not_evicted_or_counted(ckpt_path):
    base = time.time() - 1000
    for i, step in enumerate((20, 40, 60)):
        _make_export(ckpt_path, f"global_step{step}_hf", base + i)

    removed = ckpt_utils.rotate_hf_checkpoints(str(ckpt_path), "global_step60", max_num=3)

    assert removed == []
    assert _names(ckpt_path) == ["global_step20_hf", "global_step40_hf", "global_step60_hf"]


def test_no_limit_when_max_num_is_none(ckpt_path):
    base = time.time() - 1000
    for i, step in enumerate((20, 40, 60)):
        _make_export(ckpt_path, f"global_step{step}_hf", base + i)

    removed = ckpt_utils.rotate_hf_checkpoints(str(ckpt_path), "global_step80", max_num=None)

    assert removed == []
    assert len(_names(ckpt_path)) == 3


def test_no_limit_when_max_num_is_zero_or_negative(ckpt_path):
    base = time.time() - 1000
    for i, step in enumerate((20, 40, 60)):
        _make_export(ckpt_path, f"global_step{step}_hf", base + i)

    removed_zero = ckpt_utils.rotate_hf_checkpoints(str(ckpt_path), "global_step80", max_num=0)
    removed_negative = ckpt_utils.rotate_hf_checkpoints(str(ckpt_path), "global_step80", max_num=-1)

    assert removed_zero == []
    assert removed_negative == []
    assert len(_names(ckpt_path)) == 3


def test_missing_ckpt_path_is_noop(tmp_path):
    removed = ckpt_utils.rotate_hf_checkpoints(str(tmp_path / "does-not-exist"), "global_step20", max_num=3)

    assert removed == []
