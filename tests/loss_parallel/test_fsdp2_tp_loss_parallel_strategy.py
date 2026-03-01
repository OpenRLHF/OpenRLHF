from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch.nn as nn

PROJECT_ROOT = next(path for path in Path(__file__).resolve().parents if (path / "pyproject.toml").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _install_flash_attn_stubs_if_missing() -> None:
    if importlib.util.find_spec("flash_attn") is not None:
        return

    flash_attn_module = types.ModuleType("flash_attn")
    flash_attn_module.__path__ = []  # type: ignore[attr-defined]

    bert_padding_module = types.ModuleType("flash_attn.bert_padding")
    bert_padding_module.index_first_axis = lambda *args, **kwargs: None
    bert_padding_module.pad_input = lambda *args, **kwargs: None
    bert_padding_module.rearrange = lambda *args, **kwargs: None
    bert_padding_module.unpad_input = lambda *args, **kwargs: (None, None, None, None, None)

    utils_module = types.ModuleType("flash_attn.utils")
    utils_module.__path__ = []  # type: ignore[attr-defined]

    distributed_module = types.ModuleType("flash_attn.utils.distributed")
    distributed_module.all_gather = lambda *args, **kwargs: None

    sys.modules["flash_attn"] = flash_attn_module
    sys.modules["flash_attn.bert_padding"] = bert_padding_module
    sys.modules["flash_attn.utils"] = utils_module
    sys.modules["flash_attn.utils.distributed"] = distributed_module


_install_flash_attn_stubs_if_missing()

from openrlhf.utils.fsdp2 import strategy as fsdp2_strategy_module

FSDP2Strategy = fsdp2_strategy_module.FSDP2Strategy


def _build_strategy_args(*, tp_size: int, loss_parallel: bool) -> SimpleNamespace:
    return SimpleNamespace(
        fsdp2_cp_size=1,
        fsdp2_tp_size=tp_size,
        fsdp2_tp_loss_parallel=loss_parallel,
        param_dtype="fp32",
        fsdp2_cpu_offload=False,
        fsdp2_reshard_after_forward=True,
        fsdp2_tp_sequence_parallel=False,
        fsdp2_enable_sleep=False,
    )


@pytest.mark.unit
def test_fsdp2_tp_loss_parallel_requires_tp_size_gt_one() -> None:
    with pytest.raises(ValueError, match="--fsdp2_tp_loss_parallel requires --fsdp2_tp_size > 1"):
        FSDP2Strategy(args=_build_strategy_args(tp_size=1, loss_parallel=True))


@pytest.mark.unit
@pytest.mark.parametrize("loss_parallel", [False, True])
def test_apply_parallelism_forwards_shard_logits(monkeypatch: pytest.MonkeyPatch, loss_parallel: bool) -> None:
    strategy = FSDP2Strategy(args=_build_strategy_args(tp_size=2, loss_parallel=loss_parallel))
    strategy.fsdp2_dp_size = 1
    strategy.mesh = {"tp": "tp-mesh"}

    model = nn.Linear(4, 4, bias=False)
    sentinel = object()
    apply_tp_calls = []

    def _fake_apply_tensor_parallel(model_obj, tp_mesh, sequence_parallel, validate, shard_logits):
        apply_tp_calls.append(
            {
                "model": model_obj,
                "tp_mesh": tp_mesh,
                "sequence_parallel": sequence_parallel,
                "validate": validate,
                "shard_logits": shard_logits,
            }
        )
        return model_obj

    monkeypatch.setattr(fsdp2_strategy_module, "apply_tensor_parallel", _fake_apply_tensor_parallel)
    monkeypatch.setattr(strategy, "_apply_fsdp", lambda model_obj, force_cpu_offload=False: sentinel)

    output = strategy.apply_parallelism(model)
    assert output is sentinel
    assert len(apply_tp_calls) == 1
    assert apply_tp_calls[0]["model"] is model
    assert apply_tp_calls[0]["tp_mesh"] == "tp-mesh"
    assert apply_tp_calls[0]["sequence_parallel"] is False
    assert apply_tp_calls[0]["validate"] is True
    assert apply_tp_calls[0]["shard_logits"] is loss_parallel


@pytest.mark.unit
def test_apply_parallelism_skips_tp_when_tp_size_is_one(monkeypatch: pytest.MonkeyPatch) -> None:
    strategy = FSDP2Strategy(args=_build_strategy_args(tp_size=1, loss_parallel=False))
    strategy.fsdp2_dp_size = 1
    strategy.mesh = {"tp": "tp-mesh"}

    model = nn.Linear(4, 4, bias=False)
    sentinel = object()

    def _unexpected_apply_tensor_parallel(*args, **kwargs):
        raise AssertionError("apply_tensor_parallel must not be called when fsdp2_tp_size == 1")

    monkeypatch.setattr(fsdp2_strategy_module, "apply_tensor_parallel", _unexpected_apply_tensor_parallel)
    monkeypatch.setattr(strategy, "_apply_fsdp", lambda model_obj, force_cpu_offload=False: sentinel)

    output = strategy.apply_parallelism(model)
    assert output is sentinel
