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

from openrlhf.utils.fsdp2.tp.loss_parallel import compute_token_log_probs_sharded

_models_utils_path = PROJECT_ROOT / "openrlhf" / "models" / "utils.py"
_models_utils_spec = importlib.util.spec_from_file_location("openrlhf_models_utils_local", _models_utils_path)
assert _models_utils_spec is not None and _models_utils_spec.loader is not None
_models_utils = importlib.util.module_from_spec(_models_utils_spec)
_models_utils_spec.loader.exec_module(_models_utils)
compute_token_log_probs_api = _models_utils.compute_token_log_probs

WORLD_SIZE = 2
IGNORE_INDEX = -100
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


def _reference_token_log_probs(
    full_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    scaled_logits = full_logits.float()
    if temperature != 1.0:
        scaled_logits = scaled_logits / temperature

    full_log_probs = F.log_softmax(scaled_logits, dim=-1)
    token_log_probs = torch.zeros_like(labels, dtype=scaled_logits.dtype)
    valid_mask = labels != ignore_index
    if valid_mask.any():
        token_log_probs[valid_mask] = full_log_probs[valid_mask].gather(
            dim=-1,
            index=labels[valid_mask].unsqueeze(-1),
        ).squeeze(-1)
    return token_log_probs


def _worker_compute_token_log_probs(rank: int, world_size: int, port: int) -> None:
    _init_process_group(rank, world_size, port)
    try:
        mesh = _make_device_mesh(world_size)
        full_logits = _build_reference_full_logits(offset=-0.25)
        local_logits = _slice_local_logits(full_logits, rank, requires_grad=True)
        sharded_logits = _make_sharded_logits(local_logits, mesh)
        labels = torch.tensor(
            [
                [0, 3, IGNORE_INDEX],
                [4, 1, 2],
            ],
            dtype=torch.long,
        )

        temperature = 0.7
        sharded_log_probs = compute_token_log_probs_sharded(
            sharded_logits,
            labels,
            temperature=temperature,
            ignore_index=IGNORE_INDEX,
        )
        expected_log_probs = _reference_token_log_probs(
            full_logits,
            labels,
            temperature=temperature,
            ignore_index=IGNORE_INDEX,
        )
        assert_close(sharded_log_probs, expected_log_probs, rtol=RTOL, atol=ATOL)

        labels_for_api = labels.clone()
        labels_for_api[labels_for_api == IGNORE_INDEX] = 0
        api_log_probs = compute_token_log_probs_api(full_logits.clone(), labels_for_api, temperature)
        valid_mask = labels != IGNORE_INDEX
        assert_close(sharded_log_probs[valid_mask], api_log_probs[valid_mask], rtol=RTOL, atol=ATOL)

        upstream = torch.linspace(0.3, 1.3, steps=sharded_log_probs.numel(), dtype=torch.float32).view_as(sharded_log_probs)
        (sharded_log_probs * upstream).sum().backward()

        dense_ref = full_logits.clone().detach().requires_grad_(True)
        dense_ref_log_probs = _reference_token_log_probs(
            dense_ref,
            labels,
            temperature=temperature,
            ignore_index=IGNORE_INDEX,
        )
        (dense_ref_log_probs * upstream).sum().backward()
        expected_local_grad = _slice_local_dense_grad(dense_ref.grad, rank)
        assert_close(local_logits.grad, expected_local_grad, rtol=RTOL, atol=ATOL)
    finally:
        dist.destroy_process_group()


@pytest.mark.unit
def test_compute_token_log_probs_sharded() -> None:
    _run_distributed_test(_worker_compute_token_log_probs)
