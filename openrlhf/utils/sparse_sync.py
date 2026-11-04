"""Implementation of SparseRL-Sync, arXiv 2605.07330
"""

from contextlib import contextmanager

import torch

_INT32_MAX = 2**31 - 1


def index_dtype(numel: int) -> torch.dtype:
    return torch.int64 if numel > _INT32_MAX else torch.int32


@torch.no_grad()
def diff_tensor(curr: torch.Tensor, prev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert curr.shape == prev.shape and curr.dtype == prev.dtype
    int_dtype = {2: torch.int16, 4: torch.int32, 1: torch.int8, 8: torch.int64}[curr.element_size()]
    curr_flat = curr.contiguous().view(-1)
    mask = curr_flat.view(int_dtype) != prev.contiguous().view(-1).view(int_dtype)
    indices = mask.nonzero(as_tuple=False).view(-1)
    return indices.to(index_dtype(curr_flat.numel())), curr_flat[mask]


@torch.no_grad()
def apply_diff(dst: torch.Tensor, indices: torch.Tensor, values: torch.Tensor) -> None:
    dst.view(-1)[indices.long()] = values


@contextmanager
def sparse_load_context():
    """Patch Tensor.copy_ so load_weights only overwrites non-NaN elements. 
    This is a hack, will need vllm support for delta sync to get rid of this
    """
    orig_copy_ = torch.Tensor.copy_

    def _sparse_copy(dst, src, *args, **kwargs):
        if isinstance(src, torch.Tensor) and src.is_floating_point():
            mask = ~torch.isnan(src)
            dst[mask] = src[mask].to(dst.dtype)
            return dst
        return orig_copy_(dst, src, *args, **kwargs)

    torch.Tensor.copy_ = _sparse_copy
    try:
        yield
    finally:
        torch.Tensor.copy_ = orig_copy_
