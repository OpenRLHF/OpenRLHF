import torch
import torch.distributed as dist
from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
from flash_attn.utils.distributed import all_gather
from torch.distributed.tensor import DTensor

RING_ATTN_GROUP = None
RING_ATTN_PAD_MULTIPLE = 1


def set_ring_attn_group(group):
    global RING_ATTN_GROUP
    RING_ATTN_GROUP = group


def get_ring_attn_group():
    return RING_ATTN_GROUP


def set_ring_attn_pad_multiple(multiple: int) -> None:
    """Set packed-stream padding multiple for TP/SP compatibility.

    This controls extra padding applied in packing mode to make the packed token
    stream length divisible by:
      - `tp_size` when ring attention (CP) is disabled
      - `cp_size * tp_size` when ring attention (CP) is enabled

    We implement this by padding the *global* packed stream to a multiple of
    `cp_size * RING_ATTN_PAD_MULTIPLE` before slicing per CP rank.
    """
    global RING_ATTN_PAD_MULTIPLE
    if multiple is None:
        multiple = 1
    multiple = int(multiple)
    if multiple < 1:
        raise ValueError(f"multiple must be >= 1, got {multiple}")
    RING_ATTN_PAD_MULTIPLE = multiple


def reset_ring_attn_position_ids(start, end, packed_seq_lens):
    """
    Calculate position ids for packed_seq_ids[start:end].
    For example, if the packed_seq_lens is [3, 2, 4, 1], start=2, end=8,
    the position ids will be [2, 0, 1, 0, 1, 2].

    Args:
        start: the start position
        end: the end position
        packed_seq_lens: the sequence lengths of packed sequences
    """
    position_ids = torch.zeros((1, end - start), dtype=torch.long, device=torch.cuda.current_device())
    offset = 0
    for seqlen in packed_seq_lens:
        seq_start = max(offset, start)
        seq_end = min(offset + seqlen, end)
        if seq_start < seq_end:
            position_ids[0, seq_start - start : seq_end - start] = torch.arange(seq_start - offset, seq_end - offset)

        offset += seqlen
        if offset >= end:
            break
    return position_ids


def update_ring_attn_params(cu_seqlens):
    """
    Calculate the cu_seqlens for the current forward pass and pass the value to
    the substituted ring_flash_attn.

    Note that total_seq_len may be larger than the sum of packed_seq_lens because of padding.
    """
    assert RING_ATTN_GROUP is not None

    from ring_flash_attn import update_ring_flash_attn_params

    update_ring_flash_attn_params(cu_seqlens, RING_ATTN_GROUP)


def get_tensor_in_current_ring_attn_rank(tensors: list[torch.Tensor] | torch.Tensor, ring_attn_group, pad_id):
    """
    Deal with padding and slice the tensor to current ring_attn_rank.
    Args:
        tensors: Each tensor shaped (batch, seqlen) or (1, total_seqs)
        ring_attn_group: Ring attention group
        pad_id: Padding id
    Returns:
        Processed tensor
    """
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]
    ring_attn_rank = dist.get_rank(group=ring_attn_group)
    ring_attn_size = dist.get_world_size(group=ring_attn_group)
    if tensors[0].dim() != 2 or tensors[0].shape[0] != 1:
        raise ValueError(
            "Packing mode expects tensors shaped (1, total_tokens) before CP slicing; "
            f"got shape={tuple(tensors[0].shape)}"
        )
    seqlen = tensors[0].shape[-1]
    total_seq_len = tensors[0].numel()
    pad_multiple = ring_attn_size * max(1, int(RING_ATTN_PAD_MULTIPLE))
    ring_attn_pad_len = (pad_multiple - seqlen % pad_multiple) % pad_multiple
    output_tensors = []
    for tensor in tensors:
        if tensor.numel() != total_seq_len:
            raise ValueError(f"tensor.numel() {tensor.numel()} != total_seq_len {total_seq_len}")
        tensor = torch.nn.functional.pad(tensor, (0, ring_attn_pad_len), value=pad_id)
        local_seq_len = tensor.shape[-1] // ring_attn_size
        if RING_ATTN_PAD_MULTIPLE > 1 and local_seq_len % RING_ATTN_PAD_MULTIPLE != 0:
            raise AssertionError(
                f"local_seq_len ({local_seq_len}) must be divisible by RING_ATTN_PAD_MULTIPLE ({RING_ATTN_PAD_MULTIPLE})"
            )
        start, end = ring_attn_rank * local_seq_len, (ring_attn_rank + 1) * local_seq_len
        tensor = tensor[:, start:end]
        output_tensors.append(tensor)
    if len(output_tensors) == 1:
        output_tensors = output_tensors[0]
    return output_tensors, ring_attn_pad_len


def unpad_and_slice_tensor(sequences, attention_mask, ring_attn_group):
    """
    Unpad and slice tensor for distributed training with ring attention.

    This function performs several operations:
    1. Removes padding, unpads sequences from (batch, seqlen) to (1, total_seqs)
    2. Pads the packed stream for parallelism:
       - With ring_attn_group: pad to be divisible by (cp_size * tp_pad_multiple) then slice per CP rank
       - Without ring_attn_group: optionally pad to be divisible by tp_pad_multiple (for Sequence Parallel)
    3. Slices the sequences for the current ring_attn_rank

    Example:
        >>> # Input sequences shape: (batch=2, seqlen=4)
        >>> sequences = [[1, 2, 3, 0], [4, 5, 0, 0]]  # 0 is padding
        >>> attention_mask = [[1, 1, 1, 0], [1, 1, 0, 0]]
        >>> # After unpad:
        >>> # sequences: [1, 2, 3, 4, 5]  # shape (1, total_seqs=5)
        >>> # If ring_attn_group size is 2, it will pad to length 6
        >>> # Then slice for current rank (e.g., rank 0 gets [1,2,3], rank 1 gets [4,5,0])

    Args:
        sequences: Input sequences tensor of shape (batch, seqlen)
        attention_mask: Attention mask tensor for the sequences
        ring_attn_group: Ring attention group for distributed processing

    Returns:
        tuple: Processed sequences and related tensors for ring attention
    """
    rolled_sequences = torch.roll(sequences, shifts=-1, dims=1)
    sequences, indices, cu_seqlens, _, _ = unpad_input(sequences.unsqueeze(-1), attention_mask)
    sequences = sequences.transpose(0, 1)  # (1, total_seqs)
    rolled_sequences = index_first_axis(
        rearrange(rolled_sequences.unsqueeze(-1), "b s ... -> (b s) ..."), indices
    ).transpose(
        0, 1
    )  # (1, total_seqs)
    position_ids = torch.clip(torch.cumsum(attention_mask, dim=-1) - 1, min=0, max=None)
    position_ids = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(
        0, 1
    )  # (1, total_seqs)
    ring_attn_pad_len = 0
    if ring_attn_group is not None:
        (sequences, position_ids, rolled_sequences), ring_attn_pad_len = get_tensor_in_current_ring_attn_rank(
            [sequences, position_ids, rolled_sequences], ring_attn_group, 0
        )
        cu_seqlens[-1] += ring_attn_pad_len
        update_ring_attn_params(cu_seqlens)
    elif RING_ATTN_PAD_MULTIPLE > 1:
        # TP+SP requires seq length divisible by TP size. When ring-attn is disabled
        # (cp_group is None), we still pad the packed stream so Sequence Parallel can
        # evenly shard the sequence dimension across TP ranks.
        pad_multiple = RING_ATTN_PAD_MULTIPLE
        seqlen = sequences.shape[-1]
        ring_attn_pad_len = (pad_multiple - seqlen % pad_multiple) % pad_multiple
        if ring_attn_pad_len:
            sequences = torch.nn.functional.pad(sequences, (0, ring_attn_pad_len), value=0)
            rolled_sequences = torch.nn.functional.pad(rolled_sequences, (0, ring_attn_pad_len), value=0)
            position_ids = torch.nn.functional.pad(position_ids, (0, ring_attn_pad_len), value=0)
        if sequences.shape[-1] % pad_multiple != 0:
            raise AssertionError(
                f"packed_seq_len ({sequences.shape[-1]}) must be divisible by RING_ATTN_PAD_MULTIPLE ({pad_multiple})"
            )
    return sequences, position_ids, rolled_sequences, ring_attn_pad_len, indices


def gather_and_pad_tensor(tensor, ring_attn_group, ring_attn_pad_len, indices, batch, seqlen):
    """
    Gather and pad tensor data (such as logits, log_probs, etc.).

    Example:
        >>> # Input tensor from each rank (shape: (1, local_seq_len))
        >>> # Rank 0: [1, 2, 3]
        >>> # Rank 1: [4, 5, 0]  # 0 is padding
        >>> # After all_gather:
        >>> # tensor: [1, 2, 3, 4, 5, 0]  # shape (1, total_seqs=6)
        >>> # After removing padding (ring_attn_pad_len=1):
        >>> # tensor: [1, 2, 3, 4, 5]  # shape (1, total_seqs=5)
        >>> # After pad_input with original indices:
        >>> # tensor: [[1, 2, 3, 0], [4, 5, 0, 0]]  # shape (batch=2, seqlen=4)

    Args:
        tensor: Input tensor, can be logits, log_probs, etc.
        ring_attn_group: Ring attention group
        ring_attn_pad_len: Padding length
        indices: Indices
        batch: Batch size
        seqlen: Sequence length

    Returns:
        Padded tensor
    """
    is_dtensor = isinstance(tensor, DTensor)
    if is_dtensor:
        mesh = tensor.device_mesh
        placements = tensor.placements
        tensor = tensor.to_local()

    if tensor.dim() < 2 or tensor.shape[0] != 1:
        raise ValueError(
            "Packing mode expects tensors shaped (1, total_tokens, ...) before CP gather; "
            f"got shape={tuple(tensor.shape)}"
        )

    # Convert from packed-stream format (1, total_tokens, ...) to flash-attn format (total_tokens, ...).
    # For scalar-per-token tensors (1, total_tokens), pad_input expects at least 2 dims, so we
    # represent it as (total_tokens, 1) and squeeze at the end.
    is_scalar_per_token = tensor.dim() == 2
    packed = tensor.squeeze(0)  # (total_tokens, ...) or (total_tokens,)
    if packed.dim() == 1:
        packed = packed.unsqueeze(-1)  # (total_tokens, 1)

    if ring_attn_group is not None:
        packed = all_gather(packed, ring_attn_group)
    if ring_attn_pad_len > 0:
        packed = packed[:-ring_attn_pad_len]

    out = pad_input(packed, indices, batch, seqlen)
    if is_scalar_per_token:
        out = out.squeeze(-1)

    if is_dtensor:
        # Preserve TP sharding metadata (typically Shard(-1) on vocab dim).
        return DTensor.from_local(out, mesh, placements, run_check=False)
    return out
