import torch
import torch.distributed as dist
from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
from flash_attn.utils.distributed import all_gather

RING_ATTN_GROUP = None


def set_ring_attn_group(group):
    global RING_ATTN_GROUP
    RING_ATTN_GROUP = group


def get_ring_attn_group():
    return RING_ATTN_GROUP


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
    seqlen = tensors[0].shape[-1]
    total_seq_len = tensors[0].numel()
    ring_attn_pad_len = (ring_attn_size - seqlen % ring_attn_size) % ring_attn_size
    output_tensors = []
    for tensor in tensors:
        if tensor.numel() != total_seq_len:
            raise ValueError(f"tensor.numel() {tensor.numel()} != total_seq_len {total_seq_len}")
        tensor = torch.nn.functional.pad(tensor, (0, ring_attn_pad_len), value=pad_id)
        local_seq_len = tensor.numel() // ring_attn_size
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
    2. Adapts to ring_attn_group, pads sequences to be divisible by ring_attn_group
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
    if ring_attn_group is not None:
        tensor = all_gather(tensor.transpose(0, 1), ring_attn_group).transpose(0, 1)  # (1, total_seqs)
        if ring_attn_pad_len > 0:
            tensor = tensor[:, :-ring_attn_pad_len]
    tensor = pad_input(tensor.transpose(0, 1), indices, batch, seqlen).squeeze(-1)  # (batch, seqlen)
    return tensor
