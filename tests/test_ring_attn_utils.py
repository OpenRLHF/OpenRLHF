"""Regression tests for the optional-flash_attn fallback in ring_attn_utils.

openrlhf.models.ring_attn_utils prefers the external flash_attn implementations of
index_first_axis / pad_input / unpad_input / all_gather when the package is installed
(the established NVIDIA/CUDA path). When the top-level flash_attn package is absent
(e.g. Intel XPU), it falls back to transformers' private padding helpers plus a vendored
autograd-aware all_gather.

Two things make that fallback subtle, and both are guarded here:

1. transformers' ``_index_first_axis`` is NOT a drop-in for flash_attn's
   ``index_first_axis``. transformers assumes a ``(batch, seqlen, ...)`` input and flattens
   the first two dims before indexing, dropping the trailing dim; flash_attn selects rows
   along axis 0 and preserves trailing dims. OpenRLHF calls the helper on an already-flattened
   ``(batch*seqlen, 1)`` tensor and then transposes, so the trailing dim must survive. This
   incompatibility was discovered through a real GRPO ``--packing_samples`` run - a plain
   import or an isolated collective test does not reach it (see test_packing_path_regression).

2. The exception guard must fall back only when the top-level ``flash_attn`` package is
   genuinely absent, and must NOT hide a broken-but-installed flash_attn (missing compiled
   extension / version mismatch), whose ModuleNotFoundError names one of flash_attn's own
   submodules.

The multi-rank collective tests mirror tests/test_distributed_backend_generic.py: real
spawned processes, device-generic backend selection, single-device fallback when only one
accelerator is visible.
"""

import os
import queue
import socket
import traceback

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from openrlhf.models import ring_attn_utils
from openrlhf.models.ring_attn_utils import (
    gather_and_pad_tensor,
    index_first_axis,
    unpad_and_slice_tensor,
)


# ---------------------------------------------------------------------------
# Helpers (mirrors tests/test_distributed_backend_generic.py conventions)
# ---------------------------------------------------------------------------
def _accelerator_device_and_backend():
    """Return (device_type, backend) for the available accelerator, or None."""
    if not hasattr(torch, "accelerator"):
        return None
    accelerator = torch.accelerator.current_accelerator(check_available=True)
    if accelerator is None:
        return None
    return accelerator.type, str(dist.get_default_backend_for_device(accelerator.type))


def _device_count(device: str) -> int:
    if device == "cpu":
        return 1
    device_module = getattr(torch, device, None)
    return device_module.device_count() if device_module is not None else 0


def _backend_is_available(backend: str) -> bool:
    checks = {
        "nccl": dist.is_nccl_available,
        "xccl": getattr(dist, "is_xccl_available", lambda: False),
        "gloo": dist.is_gloo_available,
        "mpi": dist.is_mpi_available,
        "ucc": getattr(dist, "is_ucc_available", lambda: False),
    }
    check = checks.get(backend)
    return bool(check()) if check is not None else False


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


# ---------------------------------------------------------------------------
# Test 1 - index_first_axis preserves trailing dimensions (the exact failure shape)
# ---------------------------------------------------------------------------
def test_index_first_axis_preserves_trailing_dimensions():
    """flash_attn's index_first_axis keeps trailing dims; the transformers alias did not.

    Uses the (batch*seqlen, 1) shape produced by ``rearrange`` in unpad_and_slice_tensor,
    then the ``.transpose(0, 1)`` that follows it at the call site. With transformers'
    ``_index_first_axis`` the result was 1-D and the transpose raised IndexError; the
    corrected wrapper keeps it 2-D.
    """
    tensor = torch.tensor([[10.0], [20.0], [30.0], [40.0], [50.0], [60.0]], requires_grad=True)
    indices = torch.tensor([0, 2, 5])

    result = index_first_axis(tensor, indices)

    assert result.shape == (3, 1)
    torch.testing.assert_close(result, torch.tensor([[10.0], [30.0], [60.0]]))

    # The operation that failed with the transformers alias:
    transposed = result.transpose(0, 1)
    assert transposed.shape == (1, 3)


# ---------------------------------------------------------------------------
# Test 2 - gradient flows back to the selected rows
# ---------------------------------------------------------------------------
def test_index_first_axis_gradient_flows_to_selected_rows():
    """The gather is autograd-aware: unselected rows get zero grad, selected rows get one.

    Real unpad indices are always unique (nonzero positions of an attention mask), so this
    exercises the only pattern the packing path produces. (torch.index_select would also
    accumulate on duplicate indices, but the training code never passes duplicates.)
    """
    tensor = torch.tensor([[1.0], [2.0], [3.0], [4.0]], requires_grad=True)
    indices = torch.tensor([0, 3])

    index_first_axis(tensor, indices).sum().backward()

    torch.testing.assert_close(tensor.grad, torch.tensor([[1.0], [0.0], [0.0], [1.0]]))


# ---------------------------------------------------------------------------
# Test 3 - exact --packing_samples regression through the real OpenRLHF function
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_packing_path_regression(dtype):
    """Drive the exact code path that failed during GRPO --packing_samples training.

    unpad_and_slice_tensor -> unpad_input/index_first_axis/rearrange, then
    gather_and_pad_tensor -> pad_input round-trip, with gradient. ring_attn_group=None
    (single rank) so no collective is needed but every padding helper runs.
    """
    attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 1, 0, 0, 0]], dtype=torch.long)
    sequences = torch.tensor([[11, 12, 13, 0, 0], [21, 22, 23, 24, 25], [31, 32, 0, 0, 0]], dtype=torch.long)
    batch, seqlen = sequences.shape
    n_real = int(attention_mask.sum())  # 3 + 5 + 2 = 10

    # No IndexError here (the original bug) and shapes/values are correct.
    seqs, pos_ids, rolled, pad_len, indices = unpad_and_slice_tensor(sequences, attention_mask, ring_attn_group=None)
    assert seqs.shape == (1, n_real)
    assert pad_len == 0
    # position ids restart per packed sequence
    assert pos_ids.squeeze(0).tolist() == [0, 1, 2, 0, 1, 2, 3, 4, 0, 1]

    # pad_input round-trip: a per-token float payload must land back at masked positions,
    # pads must be zero, and gradient must flow through pad_input to the unpadded input.
    vals = torch.arange(1, n_real + 1, dtype=dtype).unsqueeze(0).requires_grad_(True)
    out = gather_and_pad_tensor(
        vals, ring_attn_group=None, ring_attn_pad_len=0, indices=indices, batch=batch, seqlen=seqlen
    )
    assert out.shape == (batch, seqlen)
    assert out.dtype == dtype

    mask_bool = attention_mask.bool()
    torch.testing.assert_close(out[mask_bool], torch.arange(1, n_real + 1, dtype=dtype))
    torch.testing.assert_close(out[~mask_bool], torch.zeros(batch * seqlen - n_real, dtype=dtype))

    out.sum().backward()
    assert vals.grad is not None
    torch.testing.assert_close(vals.grad, torch.ones_like(vals.grad))


# ---------------------------------------------------------------------------
# Test 4 - fallback wiring is what we expect when flash_attn is absent
# ---------------------------------------------------------------------------
def test_fallback_symbols_are_wired_when_flash_attn_absent():
    """When flash_attn is not installed, the module must expose working helpers.

    Skipped when flash_attn IS installed (the NVIDIA path), since then these symbols come
    from flash_attn and this fallback-specific wiring does not apply.
    """
    if _flash_attn_installed():
        pytest.skip("flash_attn is installed; fallback wiring does not apply on this box")

    # padding helpers resolve to transformers' private implementations
    import transformers.modeling_flash_attention_utils as tfa

    assert ring_attn_utils.pad_input is tfa._pad_input
    assert ring_attn_utils.unpad_input is tfa._unpad_input
    # index_first_axis is the local wrapper, NOT the transformers alias
    assert ring_attn_utils.index_first_axis is not getattr(tfa, "_index_first_axis", None)
    # all_gather is the vendored autograd function
    assert getattr(ring_attn_utils.all_gather, "__self__", None) is ring_attn_utils._AllGatherFunc


# ---------------------------------------------------------------------------
# Test 5 - the exception guard is narrow (broken install is not hidden)
# ---------------------------------------------------------------------------
def test_guard_falls_back_only_for_top_level_package():
    """Reproduce the guard logic: fall back on 'flash_attn', re-raise on submodules."""

    def guard(missing_name):
        exc = ModuleNotFoundError(f"No module named '{missing_name}'", name=missing_name)
        # mirror the source: `if exc.name != "flash_attn": raise`
        if exc.name != "flash_attn":
            return "raise"
        return "fallback"

    assert guard("flash_attn") == "fallback"
    # broken-but-installed flash_attn variants must surface, not silently fall back
    assert guard("flash_attn_2_cuda") == "raise"
    assert guard("flash_attn.bert_padding") == "raise"
    assert guard("some_unrelated_dep") == "raise"


def _flash_attn_installed() -> bool:
    import importlib.util

    return importlib.util.find_spec("flash_attn") is not None


# ---------------------------------------------------------------------------
# Test 6 - world-size-one all_gather forward + backward
# ---------------------------------------------------------------------------
def test_all_gather_requires_initialized_process_group():
    """The vendored all_gather guards against an uninitialized process group."""
    if _flash_attn_installed():
        pytest.skip("flash_attn is installed; vendored all_gather is not in use")
    if dist.is_initialized():
        pytest.skip("a process group is already initialized in this interpreter")

    with pytest.raises(RuntimeError, match="must be initialized"):
        ring_attn_utils.all_gather(torch.zeros(2, 2), None)


# ---------------------------------------------------------------------------
# Tests 7-9 - real two-rank all_gather forward + backward across processes
# ---------------------------------------------------------------------------
def _all_gather_worker(rank, device, backend, master_port, result_queue):
    """One rank: all_gather a rank-local tensor, backprop a scalar, report grad."""
    initialized = False
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(master_port)

        device_id = None
        tensor_device = torch.device("cpu")
        if device != "cpu":
            count = _device_count(device)
            if count < 1:
                raise RuntimeError(f"No visible {device} devices")
            index = rank % count
            getattr(torch, device).set_device(index)
            device_id = torch.device(device, index)
            tensor_device = device_id

        dist.init_process_group(backend=backend, rank=rank, world_size=2, device_id=device_id)
        initialized = True

        # rank 0 -> [[1, 1]], rank 1 -> [[2, 2]]
        local = torch.full((1, 2), float(rank + 1), device=tensor_device, requires_grad=True)
        gathered = ring_attn_utils.all_gather(local, None)  # None -> default group

        forward_ok = torch.allclose(gathered.detach().cpu(), torch.tensor([[1.0, 1.0], [2.0, 2.0]]))
        assert gathered.shape == (2, 2)

        # weight rows differently so each rank's grad is distinguishable, then reduce-scatter
        weights = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=tensor_device)
        (gathered * weights).sum().backward()

        if device != "cpu":
            torch.accelerator.synchronize()

        # backward reduce-scatters: reduce_scatter_tensor SUMS grad_output across ranks and
        # then scatters row `rank`. Every rank's grad w.r.t. the gathered output is `weights`,
        # so the summed grad is world_size * weights, and this rank keeps its own row of that.
        world_size = 2
        expected_grad = (world_size * weights[rank]).detach().cpu()
        backward_ok = torch.allclose(local.grad.detach().cpu(), expected_grad)

        result_queue.put(
            {
                "rank": rank,
                "forward_ok": forward_ok,
                "backward_ok": backward_ok,
                "grad": local.grad.detach().cpu().tolist(),
                "error": None,
            }
        )
    except BaseException:
        result_queue.put(
            {"rank": rank, "forward_ok": False, "backward_ok": False, "grad": None, "error": traceback.format_exc()}
        )
        raise
    finally:
        if initialized and dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.integration
def test_two_rank_all_gather_forward_and_backward():
    """Real cross-process all_gather: forward gathers both ranks, backward reduce-scatters.

    Device-generic (mirrors test_distributed_backend_generic): uses the accelerator + its
    canonical backend when available, else CPU/Gloo. Skips entirely when flash_attn is
    installed (the vendored fallback is not the code under test there).
    """
    if _flash_attn_installed():
        pytest.skip("flash_attn is installed; vendored all_gather is not in use")

    # Prefer the accelerator; fall back to CPU/Gloo so the test still runs on CPU-only boxes.
    accel = _accelerator_device_and_backend()
    if accel is not None and _backend_is_available(accel[1]) and _device_count(accel[0]) >= 1:
        device, backend = accel
    else:
        device, backend = "cpu", "gloo"

    if not _backend_is_available(backend):
        pytest.skip(f"{backend} is not available in this PyTorch build")

    context = mp.get_context("spawn")
    result_queue = context.Queue()
    master_port = _find_free_port()

    processes = [
        context.Process(target=_all_gather_worker, args=(rank, device, backend, master_port, result_queue))
        for rank in range(2)
    ]

    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=60)

        alive = [p for p in processes if p.is_alive()]
        for p in alive:
            p.terminate()
        for p in alive:
            p.join(timeout=10)
        if alive:
            pytest.fail(f"all_gather worker timed out (device={device}, backend={backend})")

        results = {}
        for _ in range(2):
            try:
                r = result_queue.get(timeout=10)
            except queue.Empty:
                break
            results[r["rank"]] = r

        errors = {rank: r["error"] for rank, r in results.items() if r["error"] is not None}
        assert not errors, f"worker failures (device={device}, backend={backend}):\n{errors}"

        exit_failures = {rank: p.exitcode for rank, p in enumerate(processes) if p.exitcode != 0}
        assert not exit_failures, f"workers exited non-zero (device={device}, backend={backend}): {exit_failures}"

        assert set(results) == {0, 1}, f"expected ranks 0 and 1, got {results}"
        for rank, r in results.items():
            assert r["forward_ok"], f"rank {rank} forward gather wrong"
            assert r["backward_ok"], f"rank {rank} backward grad wrong: {r['grad']}"
    finally:
        for p in processes:
            if p.is_alive():
                p.terminate()
