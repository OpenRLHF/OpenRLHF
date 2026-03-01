from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.testing import assert_close

PROJECT_ROOT = next(path for path in Path(__file__).resolve().parents if (path / "pyproject.toml").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openrlhf.utils.fsdp2.tp.loss_parallel import _allreduce_sum

WORLD_SIZE = 2
ATOL = 1e-6
RTOL = 1e-5


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _init_process_group(rank: int, world_size: int, port: int) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )


def _run_distributed_test(worker) -> None:
    port = _find_free_port()
    mp.spawn(worker, args=(WORLD_SIZE, port), nprocs=WORLD_SIZE, join=True)


def _worker_allreduce_sum(rank: int, world_size: int, port: int) -> None:
    _init_process_group(rank, world_size, port)
    try:
        group = dist.group.WORLD

        x_sum = torch.tensor([rank + 1.0], dtype=torch.float32, requires_grad=True)
        y_sum = _allreduce_sum(x_sum, group, mean_grad=False)
        assert_close(y_sum, torch.tensor([3.0], dtype=torch.float32), rtol=RTOL, atol=ATOL)
        y_sum.sum().backward()
        assert_close(x_sum.grad, torch.tensor([float(world_size)], dtype=torch.float32), rtol=RTOL, atol=ATOL)

        x_mean = torch.tensor([rank + 1.0], dtype=torch.float32, requires_grad=True)
        y_mean = _allreduce_sum(x_mean, group, mean_grad=True)
        assert_close(y_mean, torch.tensor([3.0], dtype=torch.float32), rtol=RTOL, atol=ATOL)
        y_mean.sum().backward()
        assert_close(x_mean.grad, torch.tensor([1.0], dtype=torch.float32), rtol=RTOL, atol=ATOL)
    finally:
        dist.destroy_process_group()


@pytest.mark.unit
def test_allreduce_sum() -> None:
    x = torch.randn(2, 3, dtype=torch.float32, requires_grad=True)
    y = _allreduce_sum(x, group=None)
    assert y.data_ptr() == x.data_ptr()
    y.sum().backward()
    assert_close(x.grad, torch.ones_like(x))

    _run_distributed_test(_worker_allreduce_sum)
