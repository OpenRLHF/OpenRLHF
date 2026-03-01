from __future__ import annotations

import importlib.util
import socket
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard
from torch.testing import assert_close

PROJECT_ROOT = next(path for path in Path(__file__).resolve().parents if (path / "pyproject.toml").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openrlhf.utils.fsdp2.tp.loss_parallel import compute_entropy_sharded

_models_utils_path = PROJECT_ROOT / "openrlhf" / "models" / "utils.py"
_models_utils_spec = importlib.util.spec_from_file_location("openrlhf_models_utils_local", _models_utils_path)
assert _models_utils_spec is not None and _models_utils_spec.loader is not None
_models_utils = importlib.util.module_from_spec(_models_utils_spec)
_models_utils_spec.loader.exec_module(_models_utils)
compute_entropy_api = _models_utils.compute_entropy

WORLD_SIZE = 2
LOCAL_VOCAB_SIZES = (3, 2)
GLOBAL_VOCAB_SIZE = sum(LOCAL_VOCAB_SIZES)
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


def _slice_local_logits(full_logits: torch.Tensor, rank: int, *, requires_grad: bool) -> torch.Tensor:
    start, end = _get_vocab_shard_bounds(rank)
    local_logits = full_logits[..., start:end].clone().detach()
    return local_logits.requires_grad_(requires_grad)


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


def _slice_local_dense_grad(dense_grad: torch.Tensor, rank: int) -> torch.Tensor:
    start, end = _get_vocab_shard_bounds(rank)
    return dense_grad[..., start:end]


def _reference_entropy(full_logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(full_logits.float(), dim=-1)
    log_probs = F.log_softmax(full_logits.float(), dim=-1)
    return -(probs * log_probs).sum(dim=-1)


def _worker_compute_entropy(rank: int, world_size: int, port: int) -> None:
    _init_process_group(rank, world_size, port)
    try:
        mesh = _make_device_mesh(world_size)
        full_logits = _build_reference_full_logits(offset=0.4)
        local_logits = _slice_local_logits(full_logits, rank, requires_grad=True)
        sharded_logits = _make_sharded_logits(local_logits, mesh)

        entropy_sharded = compute_entropy_sharded(sharded_logits)
        expected_entropy = _reference_entropy(full_logits)
        assert_close(entropy_sharded, expected_entropy, rtol=RTOL, atol=ATOL)

        api_entropy = compute_entropy_api(full_logits.clone())
        assert_close(entropy_sharded, api_entropy, rtol=RTOL, atol=ATOL)

        upstream = torch.linspace(0.1, 1.1, steps=entropy_sharded.numel(), dtype=torch.float32).view_as(entropy_sharded)
        (entropy_sharded * upstream).sum().backward()

        dense_ref = full_logits.clone().detach().requires_grad_(True)
        (_reference_entropy(dense_ref) * upstream).sum().backward()
        expected_local_grad = _slice_local_dense_grad(dense_ref.grad, rank)
        assert_close(local_logits.grad, expected_local_grad, rtol=RTOL, atol=ATOL)
    finally:
        dist.destroy_process_group()


@pytest.mark.unit
def test_compute_entropy_sharded() -> None:
    _run_distributed_test(_worker_compute_entropy)
