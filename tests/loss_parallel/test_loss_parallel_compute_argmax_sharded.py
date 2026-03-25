from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard
from torch.testing import assert_close

PROJECT_ROOT = next(path for path in Path(__file__).resolve().parents if (path / "pyproject.toml").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openrlhf.utils.fsdp2.tp.loss_parallel import compute_argmax_sharded

WORLD_SIZE = 2
LOCAL_VOCAB_SIZES = (3, 2)
GLOBAL_VOCAB_SIZE = sum(LOCAL_VOCAB_SIZES)


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


def _build_reference_full_logits(offset: float = 0.0) -> torch.Tensor:
    logits = torch.tensor(
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
    return logits + float(offset)


def _slice_local_logits(full_logits: torch.Tensor, rank: int) -> torch.Tensor:
    start, end = _get_vocab_shard_bounds(rank)
    return full_logits[..., start:end].clone().detach()


def _make_sharded_logits(local_logits: torch.Tensor, mesh: DeviceMesh) -> DTensor:
    shape = torch.Size((*local_logits.shape[:-1], GLOBAL_VOCAB_SIZE))
    stride = torch.empty(shape, device=local_logits.device).stride()
    return DTensor.from_local(
        local_logits,
        device_mesh=mesh,
        placements=[Shard(-1)],
        run_check=False,
        shape=shape,
        stride=stride,
    )


def _worker_compute_argmax(rank: int, world_size: int, port: int) -> None:
    _init_process_group(rank, world_size, port)
    try:
        mesh = _make_device_mesh(world_size)
        full_logits = _build_reference_full_logits(offset=0.05)
        local_logits = _slice_local_logits(full_logits, rank)
        sharded_logits = _make_sharded_logits(local_logits, mesh)
        mask = torch.tensor(
            [
                [True, False, True],
                [True, True, False],
            ],
            dtype=torch.bool,
        )

        argmax_sharded = compute_argmax_sharded(sharded_logits, mask)
        expected_argmax = full_logits[mask].argmax(dim=-1)
        assert_close(argmax_sharded, expected_argmax)
    finally:
        dist.destroy_process_group()


@pytest.mark.unit
def test_compute_argmax_sharded() -> None:
    _run_distributed_test(_worker_compute_argmax)


def _worker_compute_argmax_with_cross_rank_ties(rank: int, world_size: int, port: int) -> None:
    _init_process_group(rank, world_size, port)
    try:
        mesh = _make_device_mesh(world_size)
        full_logits = torch.tensor(
            [
                [
                    [1.0, 2.0, 5.0, 0.0, 5.0],
                    [0.0, -1.0, 1.0, 1.0, 0.5],
                ]
            ],
            dtype=torch.float32,
        )
        local_logits = _slice_local_logits(full_logits, rank)
        sharded_logits = _make_sharded_logits(local_logits, mesh)
        mask = torch.tensor([[True, True]], dtype=torch.bool)

        argmax_sharded = compute_argmax_sharded(sharded_logits, mask)
        expected_argmax = torch.tensor([4, 3], dtype=torch.long)
        assert_close(argmax_sharded, expected_argmax)
    finally:
        dist.destroy_process_group()


@pytest.mark.unit
def test_compute_argmax_sharded_cross_rank_tie_break() -> None:
    _run_distributed_test(_worker_compute_argmax_with_cross_rank_ties)
