from __future__ import annotations

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

from openrlhf.utils.fsdp2.tp.loss_parallel import compute_kd_loss_sharded

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


def _run_distributed_test(worker, *worker_args) -> None:
    port = _find_free_port()
    mp.spawn(worker, args=(WORLD_SIZE, port, *worker_args), nprocs=WORLD_SIZE, join=True)


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


def _reference_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    teacher_probs = F.softmax(teacher_logits.float(), dim=-1)
    student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
    cross_entropy = (teacher_probs * student_log_probs).sum(dim=-1)
    mask = (labels != ignore_index).float()
    return -(cross_entropy.view(-1) * mask.view(-1)).sum() / mask.sum()


def _worker_compute_kd_loss(rank: int, world_size: int, port: int, mode: str) -> None:
    _init_process_group(rank, world_size, port)
    try:
        mesh = _make_device_mesh(world_size)
        labels = torch.tensor(
            [
                [0, IGNORE_INDEX, 4],
                [2, 3, 1],
            ],
            dtype=torch.long,
        )

        full_student = _build_reference_full_logits(offset=-0.1)
        full_teacher = _build_reference_full_logits(offset=0.45)

        if mode in {"both_dtensor", "student_dtensor"}:
            local_student = _slice_local_logits(full_student, rank, requires_grad=True)
            student_input = _make_sharded_logits(local_student, mesh)
        else:
            local_student = None
            student_input = full_student.clone().detach().requires_grad_(True)

        local_teacher = _slice_local_logits(full_teacher, rank, requires_grad=False)
        teacher_dtensor = _make_sharded_logits(local_teacher, mesh)

        if mode == "both_dtensor":
            teacher_input = teacher_dtensor
        elif mode == "student_dtensor":
            teacher_input = full_teacher
        elif mode == "teacher_dtensor":
            teacher_input = teacher_dtensor
        else:
            raise ValueError(f"Unknown mode: {mode}")

        kd_loss_sharded = compute_kd_loss_sharded(student_input, teacher_input, labels, ignore_index=IGNORE_INDEX)

        student_ref = full_student.clone().detach().requires_grad_(True)
        kd_loss_ref = _reference_kd_loss(student_ref, full_teacher, labels, ignore_index=IGNORE_INDEX)
        assert_close(kd_loss_sharded, kd_loss_ref, rtol=RTOL, atol=ATOL)

        if mode in {"both_dtensor", "student_dtensor"}:
            kd_loss_sharded.backward()
            kd_loss_ref.backward()

            assert local_student is not None
            expected_local_grad = _slice_local_dense_grad(student_ref.grad, rank)
            assert_close(local_student.grad, expected_local_grad, rtol=RTOL, atol=ATOL)
        else:
            kd_loss_sharded.backward()

            start, end = _get_vocab_shard_bounds(rank)
            teacher_probs = F.softmax(full_teacher.float(), dim=-1)

            student_local_ref = full_student.clone().detach().requires_grad_(True)
            student_logsumexp = torch.logsumexp(student_local_ref.float(), dim=-1)
            student_logprobs_local = student_local_ref[..., start:end] - student_logsumexp.unsqueeze(-1)
            local_cross_entropy = (teacher_probs[..., start:end] * student_logprobs_local).sum(dim=-1)
            mask = (labels != IGNORE_INDEX).float()
            local_loss_ref = -(local_cross_entropy.view(-1) * mask.view(-1)).sum() / mask.sum()
            local_loss_ref.backward()

            assert_close(student_input.grad, student_local_ref.grad, rtol=RTOL, atol=ATOL)
    finally:
        dist.destroy_process_group()


@pytest.mark.unit
def test_compute_kd_loss_sharded() -> None:
    student_logits = torch.randn(2, 3, GLOBAL_VOCAB_SIZE)
    teacher_logits = torch.randn(2, 3, GLOBAL_VOCAB_SIZE)
    labels = torch.zeros(2, 3, dtype=torch.long)

    try:
        compute_kd_loss_sharded(student_logits, teacher_logits, labels, ignore_index=IGNORE_INDEX)
        raise AssertionError("compute_kd_loss_sharded should raise TypeError when both inputs are dense tensors")
    except TypeError as exc:
        if "requires logits or teacher_logits to be DTensor" not in str(exc):
            raise

    for mode in ("both_dtensor", "student_dtensor", "teacher_dtensor"):
        _run_distributed_test(_worker_compute_kd_loss, mode)
