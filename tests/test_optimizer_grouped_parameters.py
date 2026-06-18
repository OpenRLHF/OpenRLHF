"""
Tests for get_optimizer_grouped_parameters in deepspeed_utils.

Regression test for the Torch 2.10 LoRA crash: DeepSpeed removes empty param
groups from the optimizer in-place, but the LR scheduler captures base_lrs
from the original group count. Torch 2.10 uses strict=True in
LRScheduler._update_lr, so the mismatch raises on the first scheduler step.
LoRA adapter params (lora_A/lora_B) never match no_decay_name_list patterns,
so the zero-weight-decay group is always empty in LoRA runs.
"""

import types

import torch
import torch.nn as nn


def _load_fn():
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "deepspeed_utils",
        root / "openrlhf" / "utils" / "deepspeed" / "deepspeed_utils.py",
    )
    # Stub out heavy imports so the module loads without deepspeed installed.
    import sys

    sys.modules.setdefault("deepspeed", types.ModuleType("deepspeed"))
    ds_runtime = types.ModuleType("deepspeed.runtime")
    ds_zero = types.ModuleType("deepspeed.runtime.zero")
    ds_pp = types.ModuleType("deepspeed.runtime.zero.partition_parameters")
    ds_pp.ZeroParamStatus = object()
    sys.modules.setdefault("deepspeed.runtime", ds_runtime)
    sys.modules.setdefault("deepspeed.runtime.zero", ds_zero)
    sys.modules.setdefault("deepspeed.runtime.zero.partition_parameters", ds_pp)

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_utils = _load_fn()
get_optimizer_grouped_parameters = _utils.get_optimizer_grouped_parameters


class FullModel(nn.Module):
    """Both weight-decay and no-decay params present."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)  # weight (decay) + bias (no-decay)


class LoRAModel(nn.Module):
    """Simulates LoRA: only adapter params, no bias/norm — no-decay group is empty."""

    def __init__(self):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(4, 2))
        self.lora_B = nn.Parameter(torch.randn(2, 4))


def test_full_model_returns_two_groups():
    model = FullModel()
    groups = get_optimizer_grouped_parameters(model, weight_decay=0.01)
    assert len(groups) == 2
    assert groups[0]["weight_decay"] == 0.01
    assert groups[1]["weight_decay"] == 0.0


def test_lora_model_empty_group_filtered():
    """LoRA: no-decay group has no params → must be filtered out to avoid torch 2.10 crash."""
    model = LoRAModel()
    groups = get_optimizer_grouped_parameters(model, weight_decay=0.01)
    assert len(groups) == 1, "empty no-decay group must be removed"
    assert groups[0]["weight_decay"] == 0.01
    assert len(groups[0]["params"]) == 2  # lora_A and lora_B


def test_no_group_is_empty():
    """Invariant: returned groups always have at least one param."""
    for model in [FullModel(), LoRAModel()]:
        groups = get_optimizer_grouped_parameters(model, weight_decay=0.01)
        for g in groups:
            assert len(g["params"]) > 0, "empty param group slipped through"
