"""Device-generic distributed-backend integration smoke tests.

These tests verify that the canonical PyTorch distributed backend for the
selected device can initialize and perform a real two-rank all-reduce.

Backend selection is delegated to
``torch.distributed.get_default_backend_for_device``:

- CPU normally selects Gloo.
- CUDA normally selects NCCL.
- XPU normally selects XCCL.

The two-rank test gives each rank a different initial value and verifies that
both ranks receive the sum. This proves that data was exchanged across
processes rather than merely validating configuration or import behavior.

On systems with multiple accelerators, ranks are assigned to separate devices.
On systems with one accelerator, both ranks intentionally share device 0. The
single-device case validates cross-process collective operation on one physical
accelerator; it does not replace multi-device distributed validation.

Existing backend-specific regression tests remain unchanged.
"""

import os
import queue
import socket
import traceback

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _detect_device_and_backend() -> list[tuple[str, str]]:
    """Return CPU/Gloo and the available accelerator's canonical backend.

    Backend selection is delegated to PyTorch rather than being inferred from
    vendor-specific device checks.
    """
    pairs = [("cpu", str(dist.get_default_backend_for_device("cpu")))]

    if not hasattr(torch, "accelerator"):
        return pairs

    accelerator = torch.accelerator.current_accelerator(check_available=True)
    if accelerator is not None:
        backend = dist.get_default_backend_for_device(accelerator.type)
        pairs.append((accelerator.type, str(backend)))

    return pairs


def _device_count(device: str) -> int:
    """Return the visible device count for the selected device type."""
    if device == "cpu":
        return 1

    device_module = getattr(torch, device, None)
    if device_module is None:
        return 0

    return device_module.device_count()


def _backend_is_available(backend: str) -> bool:
    """Return whether a known PyTorch distributed backend is available."""
    availability_checks = {
        "nccl": dist.is_nccl_available,
        "xccl": getattr(dist, "is_xccl_available", lambda: False),
        "gloo": dist.is_gloo_available,
        "mpi": dist.is_mpi_available,
        "ucc": getattr(dist, "is_ucc_available", lambda: False),
    }

    check = availability_checks.get(backend)
    if check is None:
        return False

    return bool(check())


def _find_free_port() -> int:
    """Ask the operating system for an available local TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


DEVICE_BACKEND_PAIRS = _detect_device_and_backend()


@pytest.mark.integration
@pytest.mark.parametrize(
    "device,backend",
    DEVICE_BACKEND_PAIRS,
    ids=[f"{device}-{backend}" for device, backend in DEVICE_BACKEND_PAIRS],
)
def test_backend_is_available(device: str, backend: str) -> None:
    """The canonical backend for the selected device is available."""
    assert _backend_is_available(backend), (
        f"{backend} is PyTorch's default distributed backend for {device}, "
        "but it is not available in this PyTorch build"
    )


def _two_rank_worker(
    rank: int,
    device: str,
    backend: str,
    master_port: int,
    result_queue,
) -> None:
    """Run one rank of a real two-process all-reduce operation."""
    process_group_initialized = False

    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(master_port)

        device_id = None
        tensor_device = torch.device("cpu")

        if device != "cpu":
            count = _device_count(device)
            if count < 1:
                raise RuntimeError(f"No visible {device} devices")

            device_index = rank % count
            device_module = getattr(torch, device)
            device_module.set_device(device_index)

            device_id = torch.device(device, device_index)
            tensor_device = device_id

        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=2,
            device_id=device_id,
        )
        process_group_initialized = True

        tensor = torch.full(
            (4,),
            float(rank + 1),
            device=tensor_device,
        )

        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        if device != "cpu":
            torch.accelerator.synchronize()

        expected = torch.full(
            (4,),
            3.0,
            device=tensor.device,
        )

        result_queue.put(
            {
                "rank": rank,
                "matched": torch.allclose(tensor, expected),
                "value": tensor.cpu().tolist(),
                "error": None,
            }
        )

    except BaseException:
        result_queue.put(
            {
                "rank": rank,
                "matched": False,
                "value": None,
                "error": traceback.format_exc(),
            }
        )
        raise

    finally:
        if process_group_initialized and dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.integration
@pytest.mark.parametrize(
    "device,backend",
    DEVICE_BACKEND_PAIRS,
    ids=[f"{device}-{backend}" for device, backend in DEVICE_BACKEND_PAIRS],
)
def test_two_rank_allreduce_crosses_processes(
    device: str,
    backend: str,
) -> None:
    """Two ranks must both receive the summed value after all-reduce."""
    if not _backend_is_available(backend):
        pytest.skip(f"{backend} is not available in this PyTorch build")

    if device != "cpu" and _device_count(device) < 1:
        pytest.skip(f"No visible {device} devices")

    context = mp.get_context("spawn")
    result_queue = context.Queue()
    master_port = _find_free_port()

    processes = [
        context.Process(
            target=_two_rank_worker,
            args=(rank, device, backend, master_port, result_queue),
        )
        for rank in range(2)
    ]

    try:
        for process in processes:
            process.start()

        for process in processes:
            process.join(timeout=60)

        timed_out = [process for process in processes if process.is_alive()]
        if timed_out:
            for process in timed_out:
                process.terminate()

            for process in timed_out:
                process.join(timeout=10)

            timed_out_pids = [process.pid for process in timed_out]
            pytest.fail("Distributed worker timed out " f"(device={device}, backend={backend}, pids={timed_out_pids})")

        results = {}

        for _ in range(2):
            try:
                result = result_queue.get(timeout=10)
            except queue.Empty:
                break

            results[result["rank"]] = result

        worker_errors = {rank: result["error"] for rank, result in results.items() if result["error"] is not None}

        assert not worker_errors, (
            f"Distributed worker failures for device={device}, " f"backend={backend}:\n{worker_errors}"
        )

        exit_failures = {rank: process.exitcode for rank, process in enumerate(processes) if process.exitcode != 0}

        assert not exit_failures, (
            f"Distributed workers exited unsuccessfully " f"(device={device}, backend={backend}): {exit_failures}"
        )

        assert set(results) == {0, 1}, f"Expected results from ranks 0 and 1, got {results}"

        for rank, result in results.items():
            assert result["matched"], (
                f"Rank {rank} ended with {result['value']}; expected " "[3.0, 3.0, 3.0, 3.0] after all_reduce(SUM)"
            )

    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)

        result_queue.close()
        result_queue.join_thread()
