"""Padded <-> packed conversion for the FSDP/Automodel backend.

OpenRLHF's datasets emit `(B, S)` padded batches. Packing removes padding and
creates `(1, total_tokens)` streams plus sequence-boundary metadata. HF fallback
models consume HF FlashAttention varlen kwargs; Automodel custom models consume
THD kwargs (``qkv_format=thd`` / ``cu_seqlens`` / ``max_seqlen``) when their
selected attention backend preserves those boundaries.
"""

from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor


def unshard_dtensor(tensor: torch.Tensor) -> torch.Tensor:
    """Materialize a DTensor to a plain unsharded tensor on every rank.

    Use after model forward to gather TP-sharded outputs (logits with vocab dim
    sharded by ``ColwiseParallel(output_layouts=Shard(-1))``, or hidden states
    with sequence dim sharded by ``SequenceParallel`` in SP mode) so downstream
    loss / log-prob / entropy computations can run on plain tensors.

    No-op when not under TP/SP (input is a regular ``torch.Tensor``).
    Memory cost under TP=k: each rank holds the full unsharded tensor (∝ k× the
    sharded one); ok for activation-side tensors at training resolution.
    """
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


def is_automodel_custom_model(model: Any) -> bool:
    """Best-effort check for Automodel's native implementations.

    ``NeMoAutoModel*`` may return either a HF model (when ``force_hf`` is used or
    no native implementation exists) or a class under
    ``nemo_automodel.components.models``. Only the latter accepts THD packing
    kwargs directly.
    """
    module = type(model).__module__
    return module.startswith("nemo_automodel.components.models")


def pack_padded_batch(sequences: torch.Tensor, attention_mask: torch.Tensor, *, style: str = "hf"):
    """Convert a padded `(B, S)` batch to packed `(1, total_real_tokens)` format.

    Returns:
        packed_input_ids: `(1, total_real_tokens)` — pad tokens removed
        position_ids:     `(1, total_real_tokens)` — resets at sequence boundaries
        rolled_input_ids: `(1, total_real_tokens)` — `torch.roll(input_ids, -1)` then unpadded
        indices:          flat indices into `(B*S,)` of real tokens (for `unpack_to_padded`)
        attention kwargs:  HF FlashAttention kwargs or Automodel THD kwargs
    """
    batch, seqlen = sequences.shape
    mask = attention_mask.bool()
    indices = mask.reshape(-1).nonzero(as_tuple=False).flatten()
    seq_lens = mask.sum(dim=-1, dtype=torch.int32)
    # torch.cumsum on int32 promotes to int64; varlen attention kernels require
    # int32 sequence lengths. Cast back explicitly.
    cu_seq_lens = torch.cat(
        [torch.zeros(1, dtype=torch.int32, device=sequences.device), torch.cumsum(seq_lens, dim=0).to(torch.int32)]
    )
    max_length = seq_lens.max().item() if seq_lens.numel() > 0 else 0

    packed_ids = sequences.reshape(batch * seqlen).index_select(0, indices).unsqueeze(0)
    rolled = torch.roll(sequences, shifts=-1, dims=1)
    rolled_packed = rolled.reshape(batch * seqlen).index_select(0, indices).unsqueeze(0)

    # position_ids reset at seq boundaries (`[0,1,2, 0,1, 0,1,2,3]`).
    position_ids_full = torch.clip(torch.cumsum(attention_mask, dim=-1) - 1, min=0)
    position_ids = position_ids_full.reshape(batch * seqlen).index_select(0, indices).unsqueeze(0)

    if style == "automodel":
        fa_kwargs = {
            "qkv_format": "thd",
            "cu_seqlens": cu_seq_lens,
            "cu_seqlens_padded": cu_seq_lens,
            "max_seqlen": int(max_length),
        }
    else:
        fa_kwargs = {
            "cu_seq_lens_q": cu_seq_lens,
            "cu_seq_lens_k": cu_seq_lens,
            "max_length_q": int(max_length),
            "max_length_k": int(max_length),
        }
    return packed_ids, position_ids, rolled_packed, indices, fa_kwargs


def unpack_to_padded(packed: torch.Tensor, indices: torch.Tensor, batch: int, seqlen: int) -> torch.Tensor:
    """Inverse of `pack_padded_batch` (logits / log-probs side).

    Takes a `(1, total_real_tokens)` tensor and returns a `(B, S)` padded tensor
    using the `indices` from the original pack.
    """
    packed_values = packed.squeeze(0) if packed.dim() > 1 and packed.shape[0] == 1 else packed
    output = packed_values.new_zeros((batch * seqlen, *packed_values.shape[1:]))
    output.index_copy_(0, indices, packed_values)
    return output.view(batch, seqlen, *packed_values.shape[1:])


def _distributed_log_softmax(local_logits: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    logits_max = local_logits.amax(dim=-1, keepdim=True)
    dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=group)

    shifted_logits = local_logits - logits_max
    exp_sum = shifted_logits.exp().sum(dim=-1, keepdim=True).float()
    dist.all_reduce(exp_sum, op=dist.ReduceOp.SUM, group=group)
    return shifted_logits - exp_sum.log().to(shifted_logits.dtype)


class _DistributedLogProb(torch.autograd.Function):
    """Gather selected logprobs from TP-sharded vocab logits without all-gathering logits."""

    @staticmethod
    def forward(
        ctx,
        local_logits: torch.Tensor,
        target: torch.Tensor,
        vocab_start: int,
        vocab_end: int,
        group: dist.ProcessGroup,
        inference_only: bool,
    ) -> torch.Tensor:
        target_mask = (target < vocab_start) | (target >= vocab_end)
        local_target = (target - vocab_start).masked_fill(target_mask, 0)

        log_probs = _distributed_log_softmax(local_logits.float(), group)
        softmax = log_probs.exp()

        selected = torch.gather(log_probs, -1, local_target.unsqueeze(-1)).squeeze(-1)
        selected = selected.masked_fill(target_mask, 0.0)
        dist.all_reduce(selected, op=dist.ReduceOp.SUM, group=group)

        if not inference_only:
            ctx.save_for_backward(softmax, target_mask, local_target)
        return selected

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        softmax, target_mask, local_target = ctx.saved_tensors
        grad_input = -softmax * grad_output.unsqueeze(-1)

        if softmax.ndim == 3:
            grad_input = grad_input.contiguous()
            bsz, seq, vocab = softmax.shape
            row = torch.arange(bsz, device=softmax.device).view(-1, 1).expand(-1, seq).reshape(-1)
            col = torch.arange(seq, device=softmax.device).expand(bsz, -1).reshape(-1)
            flat_base = (row * seq + col) * vocab
            valid = ~target_mask.reshape(-1)
            flat_index = flat_base.masked_select(valid) + local_target.reshape(-1).masked_select(valid)
            grad_input.view(-1).scatter_add_(0, flat_index, grad_output.reshape(-1).masked_select(valid))
        else:
            valid = ~target_mask
            grad_input.scatter_add_(
                -1,
                local_target.unsqueeze(-1),
                (grad_output * valid).unsqueeze(-1),
            )

        return grad_input, None, None, None, None, None


def log_probs_from_vocab_parallel_logits(
    vocab_parallel_logits: DTensor,
    target: torch.Tensor,
    *,
    temperature: float = 1.0,
    inference_only: bool | None = None,
) -> torch.Tensor:
    """Compute selected token logprobs for TP-vocab-sharded DTensor logits.

    ``target`` is already aligned with ``vocab_parallel_logits`` positions. This
    mirrors OpenRLHF's local ``log_probs_from_logits(logits, rolled_ids)``
    contract and avoids materializing full-vocab logits on each TP rank.
    """
    device_mesh = vocab_parallel_logits.device_mesh
    if device_mesh.mesh_dim_names is None or "tp" not in device_mesh.mesh_dim_names:
        raise ValueError("vocab_parallel_logits must be sharded on a mesh with a 'tp' dimension")

    tp_group = device_mesh.get_group("tp")
    tp_rank = dist.get_rank(group=tp_group)
    tp_size = dist.get_world_size(group=tp_group)
    vocab_per_rank = vocab_parallel_logits.shape[-1] // tp_size
    vocab_start = vocab_per_rank * tp_rank
    vocab_end = vocab_start + vocab_per_rank

    local_logits = vocab_parallel_logits.to_local()
    if temperature != 1.0:
        local_logits = local_logits / temperature
    if inference_only is None:
        inference_only = not torch.is_grad_enabled()
    return _DistributedLogProb.apply(local_logits, target, vocab_start, vocab_end, tp_group, inference_only)
