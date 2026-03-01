from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import DeviceMesh
from torch.testing import assert_close

PROJECT_ROOT = next(path for path in Path(__file__).resolve().parents if (path / "pyproject.toml").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openrlhf.utils.fsdp2.tp.loss_parallel import _sharded_logsumexp

WORLD_SIZE = 2
LOCAL_VOCAB_SIZES = (3, 2)
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


def _make_device_mesh(world_size: int) -> DeviceMesh:
    return DeviceMesh("cpu", list(range(world_size)))


def _get_vocab_shard_bounds(rank: int) -> tuple[int, int]:
    start = sum(LOCAL_VOCAB_SIZES[:rank])
    end = start + LOCAL_VOCAB_SIZES[rank]
    return start, end


def _build_reference_full_logits() -> torch.Tensor:
    return torch.tensor(
        [
            [
                [3.00, -1.00, 0.50, 2.00, 1.00],
                [0.10, 0.20, 0.30, 0.40, 0.00],
                [-2.00, -1.50, -0.10, -0.20, -0.30],
            ],
            [
                [1.00, 2.50, 2.40, 2.60, 2.70],
                [4.00, 3.00, 2.00, 1.00, 0.00],
                [0.00, 1.00, 5.00, 4.00, 3.00],
            ],
        ],
        dtype=torch.float32,
    )


def _slice_local_logits(full_logits: torch.Tensor, rank: int, *, requires_grad: bool) -> torch.Tensor:
    start, end = _get_vocab_shard_bounds(rank)
    local_logits = full_logits[..., start:end].clone().detach()
    return local_logits.requires_grad_(requires_grad)


def _slice_local_dense_grad(dense_grad: torch.Tensor, rank: int) -> torch.Tensor:
    start, end = _get_vocab_shard_bounds(rank)
    return dense_grad[..., start:end]


def _worker_sharded_logsumexp(rank: int, world_size: int, port: int) -> None:
    _init_process_group(rank, world_size, port)
    try:
        mesh = _make_device_mesh(world_size)
        process_group = mesh.get_group()

        full_logits = _build_reference_full_logits() + 0.35
        local_logits = _slice_local_logits(full_logits, rank, requires_grad=True)

        sharded_lse_sum = _sharded_logsumexp(local_logits.detach().clone(), process_group, dim=-1, mean_grad=False)
        sharded_lse_mean = _sharded_logsumexp(local_logits, process_group, dim=-1, mean_grad=True)
        expected_lse = torch.logsumexp(full_logits, dim=-1)

        assert_close(sharded_lse_sum, expected_lse, rtol=RTOL, atol=ATOL)
        assert_close(sharded_lse_mean, expected_lse, rtol=RTOL, atol=ATOL)

        upstream = torch.linspace(0.2, 1.2, steps=sharded_lse_mean.numel(), dtype=torch.float32).view_as(sharded_lse_mean)
        (sharded_lse_mean * upstream).sum().backward()

        dense_ref = full_logits.clone().detach().requires_grad_(True)
        (torch.logsumexp(dense_ref, dim=-1) * upstream).sum().backward()
        expected_local_grad = _slice_local_dense_grad(dense_ref.grad, rank)
        assert_close(local_logits.grad, expected_local_grad, rtol=RTOL, atol=ATOL)
    finally:
        dist.destroy_process_group()


@pytest.mark.unit
def test_sharded_logsumexp() -> None:
    _run_distributed_test(_worker_sharded_logsumexp)
