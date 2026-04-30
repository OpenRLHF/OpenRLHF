"""Padded <-> packed conversion for the FSDP2 model backend.

OpenRLHF's datasets emit `(B, S)` padded batches. Packing removes padding and
creates `(1, total_tokens)` streams plus sequence-boundary metadata. The exact
kwargs depend on the selected model path:

- HF flash-attn2 consumes ``FlashAttentionKwargs``.
- AutoModel custom TE consumes THD kwargs
  (``qkv_format=thd`` / ``cu_seqlens`` / ``max_seqlen``).
"""

from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Shard


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


def _cp_chunk_indices(cp_size: int) -> list[int]:
    chunk_indices = []
    for cp_rank in range(cp_size):
        chunk_indices.append(cp_rank)
        chunk_indices.append(2 * cp_size - cp_rank - 1)
    return chunk_indices


def pad_to_cp_multiple(tensor: torch.Tensor, cp_size: int, seq_dim: int = 1, value: int | float = 0) -> torch.Tensor:
    """Right-pad a sequence tensor to DTensor CP's ``2 * cp_size`` requirement."""
    if cp_size <= 1:
        return tensor
    multiple = 2 * cp_size
    pad_len = (-tensor.shape[seq_dim]) % multiple
    if pad_len == 0:
        return tensor
    pad_shape = list(tensor.shape)
    pad_shape[seq_dim] = pad_len
    pad = tensor.new_full(pad_shape, value)
    return torch.cat([tensor, pad], dim=seq_dim)


def _restore_cp_chunks(cp_rank_chunks: list[torch.Tensor], cp_size: int, seq_dim: int) -> torch.Tensor:
    tensor_chunks = []
    for rank_chunk in cp_rank_chunks:
        tensor_chunks.extend(torch.chunk(rank_chunk, chunks=2, dim=seq_dim))

    chunks_and_indices = list(zip(tensor_chunks, _cp_chunk_indices(cp_size)))
    chunks_and_indices = sorted(chunks_and_indices, key=lambda item: item[1])
    return torch.cat([chunk for chunk, _ in chunks_and_indices], dim=seq_dim)


def cp_allgather_sequence(tensor: torch.Tensor, cp_group: dist.ProcessGroup, seq_dim: int = 1) -> torch.Tensor:
    """Gather a CP-sharded sequence tensor and undo AutoModel's load-balanced chunk order.

    AutoModel/DTensor context parallelism splits the sequence into ``2 * cp``
    chunks and gives each CP rank a front/back pair. NeMo-RL restores token
    order by gathering each rank's pair, splitting it back into two chunks, and
    sorting by the original chunk index. PPO logprobs and reward/critic values
    need this full-sequence view before applying OpenRLHF masks.
    """
    cp_size = dist.get_world_size(group=cp_group)
    if cp_size <= 1:
        return tensor
    return _AllGatherCPSequence.apply(tensor, cp_group, seq_dim)


def cp_shard_sequence(tensor: torch.Tensor, cp_group: dist.ProcessGroup, seq_dim: int = 1) -> torch.Tensor:
    """Select the local head/tail CP shard without communication.

    This mirrors the load-balanced sequence order used by
    ``torch.distributed.tensor.experimental.context_parallel``: CP rank ``r``
    owns chunks ``r`` and ``2 * cp_size - r - 1``.
    """
    cp_size = dist.get_world_size(group=cp_group)
    if cp_size <= 1:
        return tensor
    cp_rank = dist.get_rank(group=cp_group)
    chunks = torch.chunk(tensor, chunks=2 * cp_size, dim=seq_dim)
    return torch.cat([chunks[cp_rank], chunks[2 * cp_size - cp_rank - 1]], dim=seq_dim)


def cp_dtensor_full_sequence(tensor: torch.Tensor, cp_mesh, seq_dim: int = 1) -> torch.Tensor:
    """Restore a CP-local load-balanced sequence shard via DTensor autograd.

    PyTorch context parallel stores each rank's local sequence as head/tail
    chunks. NeMo-RL wraps that local tensor as a CP-sharded DTensor, gathers the
    logical sequence, and uses the gathered ``seq_index`` to recover original
    token order. Keeping the gather in DTensor autograd matters for training:
    every CP rank computes the same full-sequence loss while gradients are routed
    back to the owning local shard.
    """
    if cp_mesh is None or cp_mesh.size() <= 1:
        return tensor

    local_len = tensor.shape[seq_dim]
    if local_len % 2 != 0:
        raise ValueError(f"CP local sequence length must be even, got {local_len}.")

    cp_size = cp_mesh.size()
    try:
        cp_rank = cp_mesh.get_local_rank()
    except AttributeError:
        cp_rank = dist.get_rank(group=cp_mesh.get_group())
    chunk = local_len // 2
    global_len = local_len * cp_size
    second_chunk_idx = 2 * cp_size - cp_rank - 1
    local_seq_index = torch.cat(
        [
            torch.arange(cp_rank * chunk, (cp_rank + 1) * chunk, device=tensor.device),
            torch.arange(second_chunk_idx * chunk, (second_chunk_idx + 1) * chunk, device=tensor.device),
        ],
        dim=0,
    )
    if local_seq_index.numel() != local_len or int(local_seq_index.max().item()) >= global_len:
        raise RuntimeError("Invalid CP seq_index construction.")

    seq_index = DTensor.from_local(local_seq_index, device_mesh=cp_mesh, placements=[Shard(0)]).full_tensor()
    restore_index = torch.argsort(seq_index)
    restored = DTensor.from_local(tensor, device_mesh=cp_mesh, placements=[Shard(seq_dim)]).full_tensor()
    return restored.index_select(seq_dim, restore_index)


class _AllGatherCPSequence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, cp_group: dist.ProcessGroup, seq_dim: int):
        cp_size = dist.get_world_size(group=cp_group)
        cp_rank_chunks = [torch.empty_like(tensor) for _ in range(cp_size)]
        dist.all_gather(cp_rank_chunks, tensor, group=cp_group)

        ctx.cp_group = cp_group
        ctx.seq_dim = seq_dim
        return _restore_cp_chunks(cp_rank_chunks, cp_size, seq_dim)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        cp_size = dist.get_world_size(group=ctx.cp_group)
        cp_rank = dist.get_rank(group=ctx.cp_group)
        seq_dim = ctx.seq_dim

        chunks = torch.chunk(grad_output, chunks=2 * cp_size, dim=seq_dim)
        local = torch.cat(
            [chunks[cp_rank], chunks[2 * cp_size - cp_rank - 1]],
            dim=seq_dim,
        )
        return local, None, None


def is_automodel_custom_model(model: Any) -> bool:
    """Best-effort check for AutoModel's native implementations.

    ``NeMoAutoModel*`` may return either a HF model (when ``force_hf`` is used or
    no native implementation exists) or a class under
    ``nemo_automodel.components.models``. Only the latter accepts THD packing
    kwargs directly.

    FSDP2's ``fully_shard`` swaps ``model.__class__`` for a dynamic subclass
    whose ``__module__`` no longer starts with ``nemo_automodel`` — walk the MRO
    so the check survives that wrap.
    """
    for cls in type(model).__mro__:
        if getattr(cls, "_openrlhf_automodel_custom", False):
            return True
        if cls.__module__.startswith("nemo_automodel.components.models"):
            return True
    return False


def pack_padded_batch(sequences: torch.Tensor, attention_mask: torch.Tensor, *, style: str = "hf"):
    """Convert a padded `(B, S)` batch to packed `(1, total_real_tokens)` format.

    Returns:
        packed_input_ids: `(1, total_real_tokens)` — pad tokens removed
        position_ids:     `(1, total_real_tokens)` — resets at sequence boundaries
        rolled_input_ids: `(1, total_real_tokens)` — `torch.roll(input_ids, -1)` then unpadded
        indices:          flat indices into `(B*S,)` of real tokens (for `unpack_to_padded`)
        attn_kwargs:      HF FlashAttention kwargs or AutoModel THD kwargs
    """
    if style not in {"hf", "automodel"}:
        raise ValueError(f"Unsupported packing style: {style}")

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
        attn_kwargs = {
            "qkv_format": "thd",
            "cu_seqlens": cu_seq_lens,
            "cu_seqlens_padded": cu_seq_lens,
            "max_seqlen": int(max_length),
        }
    else:
        attn_kwargs = {
            "cu_seq_lens_q": cu_seq_lens,
            "cu_seq_lens_k": cu_seq_lens,
            "max_length_q": int(max_length),
            "max_length_k": int(max_length),
        }
    return packed_ids, position_ids, rolled_packed, indices, attn_kwargs


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

    local_logits = vocab_parallel_logits.to_local()
    global_vocab_size = vocab_parallel_logits.shape[-1]
    if global_vocab_size % tp_size == 0:
        vocab_per_rank = global_vocab_size // tp_size
        vocab_start = vocab_per_rank * tp_rank
        vocab_end = vocab_start + vocab_per_rank
    else:
        local_vocab_size = torch.tensor(local_logits.shape[-1], device=local_logits.device, dtype=torch.long)
        shard_sizes = [torch.zeros_like(local_vocab_size) for _ in range(tp_size)]
        dist.all_gather(shard_sizes, local_vocab_size, group=tp_group)
        shard_sizes = torch.stack(shard_sizes)
        vocab_start = int(shard_sizes[:tp_rank].sum().item())
        vocab_end = vocab_start + int(shard_sizes[tp_rank].item())

    if temperature != 1.0:
        local_logits = local_logits / temperature
    if inference_only is None:
        inference_only = not torch.is_grad_enabled()
    return _DistributedLogProb.apply(local_logits, target, vocab_start, vocab_end, tp_group, inference_only)
