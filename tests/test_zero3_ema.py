"""EMA updates must survive repartitioning, not just change gathered tensors."""

import importlib
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from tests.test_deepspeed_save_model import ds_module


def test_ema_updates_survive_zero3_partitioning():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            "--module",
            "tests.test_zero3_ema",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "DS_ACCELERATOR": "cpu", "OMP_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _check_distributed_ema():
    with pytest.MonkeyPatch.context() as monkeypatch:
        module = ds_module.__wrapped__(monkeypatch)
        deepspeed = module.deepspeed
        # Use real Gloo collectives without compiling the optional shared-memory accelerator.
        monkeypatch.setattr(importlib.import_module("deepspeed.comm.torch"), "build_shm_op", lambda: None)
        dist.init_process_group("gloo")
        deepspeed.init_distributed(dist_backend="gloo", auto_mpi_discovery=False)
        try:
            for stage in (2, 3):
                for dynamic_batch in (False, True):
                    args = SimpleNamespace(
                        ds=SimpleNamespace(param_dtype="bf16"),
                        train=SimpleNamespace(dynamic_batch_enable=dynamic_batch),
                    )
                    strategy = module.DeepspeedStrategy(args=args, zero_stage=stage)
                    strategy.accumulated_gradient = 2
                    config = {"train_batch_size": 2, "zero_optimization": {"stage": 3}}
                    with deepspeed.zero.Init(
                        config_dict_or_path=config, remote_device="cpu", dtype=torch.float32, enabled=stage == 3
                    ):
                        actor = torch.nn.Linear(4, 4)
                        ema = torch.nn.Linear(4, 4)
                    actor.bias.requires_grad_(False)
                    params = list(actor.parameters()) + list(ema.parameters())
                    with deepspeed.zero.GatheredParameters(params, modifier_rank=0, enabled=stage == 3):
                        with torch.no_grad():
                            actor.bias.fill_(7)
                            ema.weight.zero_()
                            ema.bias.fill_(11)

                    expected = 0.0
                    for step in range(1, 5):
                        with deepspeed.zero.GatheredParameters(actor.weight, modifier_rank=0, enabled=stage == 3):
                            with torch.no_grad():
                                actor.weight.fill_(step)
                        strategy.moving_average(actor, ema, beta=0.5, device="cpu")
                        if dynamic_batch or step % 2 == 0:
                            expected = 0.5 * step + 0.5 * expected

                        if stage == 3:
                            assert all(p.numel() == 0 and p.ds_tensor.numel() > 0 for p in params)
                        # Read back from partitions on both ranks after the update context has exited.
                        with deepspeed.zero.GatheredParameters(params, enabled=stage == 3):
                            torch.testing.assert_close(ema.weight, torch.full_like(ema.weight, expected))
                            torch.testing.assert_close(ema.bias, torch.full_like(ema.bias, 11))
                            torch.testing.assert_close(actor.weight, torch.full_like(actor.weight, step))
                            torch.testing.assert_close(actor.bias, torch.full_like(actor.bias, 7))
        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    _check_distributed_ema()
