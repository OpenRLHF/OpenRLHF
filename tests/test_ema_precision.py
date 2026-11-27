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


def test_actor_checkpoint_resumes_ema(ds_module, tmp_path):
    from unittest.mock import Mock

    def fresh():
        args = SimpleNamespace(ds=SimpleNamespace(param_dtype="bf16"), train=SimpleNamespace())
        strategy = ds_module.DeepspeedStrategy(args=args, zero_stage=2)
        strategy.accumulated_gradient = 2
        ema = nn.Linear(1, 1, bias=False).bfloat16()
        ema.weight.data.fill_(1)
        return strategy, ema

    actor = nn.Linear(1, 1, bias=False).bfloat16()
    actor.weight.data.fill_(2)
    strategy, ema = fresh()
    for _ in range(3):
        strategy.moving_average(actor, ema)

    # Exercise client-state serialization without requiring a GPU in the unit suite.
    engine = Mock(spec=ds_module.deepspeed.DeepSpeedEngine)
    engine.save_non_zero_checkpoint = True
    path = tmp_path / "client.pt"
    engine.save_checkpoint.side_effect = lambda *a, client_state, **kw: torch.save(client_state, path)
    engine.load_checkpoint.side_effect = lambda *a, **kw: (str(path), torch.load(path))
    strategy.save_ckpt(engine, str(tmp_path), client_state={"step": 3}, ema_model=ema)

    resumed, restored = fresh()
    _, metadata = resumed.load_ckpt(engine, str(tmp_path), ema_model=restored)
    assert metadata == {"step": 3}
    for _ in range(3):
        assert resumed.time_steps["ema"] == strategy.time_steps["ema"]
        assert torch.equal(restored.weight, ema.weight)
        assert torch.equal(resumed.ema_fp32[restored.weight], strategy.ema_fp32[ema.weight])
        strategy.moving_average(actor, ema)
        resumed.moving_average(actor, restored)

    torch.save({"step": 3}, path)
    legacy, restored = fresh()
    _, metadata = legacy.load_ckpt(engine, str(tmp_path), ema_model=restored)
    assert metadata == {"step": 3}
    assert not legacy.ema_fp32
    assert restored.weight.item() == 1
