"""Round-trip tests for the sparse weight sync codec.

"""

import pytest
import torch

from openrlhf.utils.sparse_sync import (
    apply_diff,
    diff_tensor,
    index_dtype,
    sparse_load_context,
)


def roundtrip(prev: torch.Tensor, curr: torch.Tensor):
    indices, values = diff_tensor(curr, prev)
    recon = prev.clone()
    apply_diff(recon, indices, values)
    assert recon.view(torch.int16 if recon.element_size() == 2 else torch.int32).equal(
        curr.view(torch.int16 if curr.element_size() == 2 else torch.int32)
    ), "reconstruction is not bitwise identical"
    return indices, values


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_sparse_perturbation(dtype):
    torch.manual_seed(0)
    prev = torch.randn(1024, 512, dtype=dtype)
    curr = prev.clone()
    flat = curr.view(-1)
    changed = torch.randperm(flat.numel())[: flat.numel() // 100]
    flat[changed] += 0.5
    indices, values = roundtrip(prev, curr)
    assert indices.numel() == values.numel()
    assert set(indices.tolist()) == set(changed.tolist())


def test_empty_diff():
    prev = torch.randn(64, 64, dtype=torch.bfloat16)
    indices, values = roundtrip(prev, prev.clone())
    assert indices.numel() == 0 and values.numel() == 0


def test_all_changed():
    prev = torch.zeros(1000, dtype=torch.bfloat16)
    curr = torch.arange(1, 1001, dtype=torch.float32).to(torch.bfloat16)
    indices, _ = roundtrip(prev, curr)
    assert indices.numel() == 1000


def test_signed_zero_and_nan_are_bytewise():
    prev = torch.tensor([0.0, float("nan"), 1.0], dtype=torch.bfloat16)
    curr = torch.tensor([-0.0, float("nan"), 1.0], dtype=torch.bfloat16)
    indices, _ = roundtrip(prev, curr)
    assert indices.tolist() == [0]


def test_noncontiguous_input():
    torch.manual_seed(1)
    base = torch.randn(128, 128, dtype=torch.bfloat16)
    prev = base.t()
    curr = prev.clone()
    curr[3, 7] += 1.0
    indices, values = diff_tensor(curr, prev.contiguous())
    recon = prev.contiguous()
    apply_diff(recon, indices, values)
    assert recon.equal(curr.contiguous())


def test_index_dtype_boundary():
    assert index_dtype(2**31 - 1) == torch.int32
    assert index_dtype(2**31) == torch.int64


def test_sparse_load_context_partial_copy():
    dst = torch.ones(4, 4, dtype=torch.bfloat16)
    src = torch.full((4, 4), float("nan"), dtype=torch.bfloat16)
    src[1, 2] = 5.0
    src[3, 0] = -2.0
    with sparse_load_context():
        dst.copy_(src)
    expected = torch.ones(4, 4, dtype=torch.bfloat16)
    expected[1, 2] = 5.0
    expected[3, 0] = -2.0
    assert dst.equal(expected)
    dst.copy_(torch.zeros(4, 4, dtype=torch.bfloat16))
    assert dst.eq(0).all()


def test_sparse_load_context_sharded_loader():
    """ Unchanged (NaN) positions must survive."""
    fused = torch.arange(64, dtype=torch.float32).reshape(8, 8).to(torch.bfloat16)
    orig = fused.clone()
    update = torch.full((4, 8), float("nan"), dtype=torch.bfloat16)
    update[2, 5] = 99.0
    with sparse_load_context():
        fused[4:8].copy_(update)
        idx = torch.zeros(3, dtype=torch.long)
        idx.copy_(torch.tensor([1, 2, 3]))
    assert fused[6, 5].item() == 99.0
    fused[6, 5] = orig[6, 5]
    assert fused.equal(orig)
    assert idx.tolist() == [1, 2, 3]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_roundtrip_cuda():
    torch.manual_seed(2)
    prev = torch.randn(4096, 1024, dtype=torch.bfloat16, device="cuda")
    curr = prev.clone()
    curr.view(-1)[::797] += 0.25
    roundtrip(prev, curr)
