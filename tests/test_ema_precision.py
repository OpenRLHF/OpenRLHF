"""A bf16/fp16 EMA model must track the model it averages."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from tests.test_deepspeed_save_model import ds_module  # noqa: F401


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_moving_average_tracks_small_updates(ds_module, dtype):
    args = SimpleNamespace(ds=SimpleNamespace(param_dtype="bf16"), train=SimpleNamespace())
    strategy = ds_module.DeepspeedStrategy(args=args, zero_stage=2)
    strategy.accumulated_gradient = 1

    # fp32 master weights drift by 1% of their scale; the model holds their rounding.
    generator = torch.Generator().manual_seed(0)
    start = (torch.randn(64, 64, generator=generator) * 0.02).to(dtype)
    direction = torch.randn(64, 64, generator=generator) * 0.02
    actor = nn.Linear(64, 64, bias=False).to(dtype)
    ema = nn.Linear(64, 64, bias=False).to(dtype)
    ema.weight.data.copy_(start)
    reference = start.double()

    beta, steps = 0.992, 1000
    for step in range(1, steps + 1):
        actor.weight.data.copy_(start.float() + step / steps * 0.01 * direction)
        strategy.moving_average(actor, ema, beta=beta, device="cpu")
        reference = beta * reference + (1 - beta) * actor.weight.data.double()

    # The EMA model keeps its dtype and is the float64 EMA rounded to it once.
    assert ema.weight.dtype == dtype
    error = (ema.weight.data.double() - reference).abs()
    rounding = (reference.to(dtype).double() - reference).abs()
    assert torch.all(error <= rounding + 2e-5 * reference.abs())
