import torch
import torch.distributed as dist
import torch.nn.functional as F
from flash_attn.utils.distributed import all_gather
from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

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


def update_ring_attn_params(packed_seq_lens, total_seq_len):
    """
    Calculate the cu_seqlens for the current forward pass and pass the value to
    the substituted ring_flash_attn.

    Note that total_seq_len may be larger than the sum of packed_seq_lens because of padding.
    """
    assert RING_ATTN_GROUP is not None
    cu_seqlens = torch.cumsum(
        torch.tensor(packed_seq_lens, device=torch.cuda.current_device(), dtype=torch.int32),
        dim=-1,
        dtype=torch.int32,
    )
    cu_seqlens = F.pad(F.pad(cu_seqlens, (1, 0), value=0), (0, 1), value=total_seq_len)

    from ring_flash_attn import update_ring_flash_attn_params
    update_ring_flash_attn_params(cu_seqlens, RING_ATTN_GROUP)

def get_slice_in_this_ring_attn_rank(total_seq_len, ring_attn_group):
    ring_attn_rank = dist.get_rank(group=ring_attn_group)
    ring_attn_size = dist.get_world_size(group=ring_attn_group)
    assert total_seq_len % ring_attn_size == 0, f"total_seq_len {total_seq_len} must be divisible by ring_attn_size {ring_attn_size}"
    local_seq_len = total_seq_len // ring_attn_size
    start, end = ring_attn_rank * local_seq_len, (ring_attn_rank + 1) * local_seq_len
    return start, end

def convert_ring_attn_params(sequences, attention_mask, packed_seq_lens, ring_attn_group, seq_dim=-1):
    # each rank within the ring group will process sequences[start:end]
    total_seq_len = sequences.numel()
    start, end = get_slice_in_this_ring_attn_rank(total_seq_len, ring_attn_group)
    sequences = sequences[:, start:end]
    attention_mask = attention_mask[:, start:end]
    position_ids = reset_ring_attn_position_ids(start, end, packed_seq_lens)
    update_ring_attn_params(packed_seq_lens, total_seq_len)
    return sequences, attention_mask, position_ids

def unpad_and_slice_tensor(sequences, attention_mask, ring_attn_group):
    """
    Unpad and slice tensor for distributed training with ring attention.

    This function performs several operations:
    1. Removes padding, unpads sequences from (batch, seqlen) to (1, total_seqs)
    2. Adapts to ring_attn_group, pads sequences to be divisible by ring_attn_group
    3. Slices the sequences for the current ring_attn_rank

    Args:
        sequences: Input sequences tensor of shape (batch, seqlen)
        attention_mask: Attention mask tensor for the sequences
        ring_attn_group: Ring attention group for distributed processing

    Returns:
        tuple: Processed sequences and related tensors for ring attention
    """
    labels = torch.roll(sequences, shifts=-1, dims=1)
    sequences, indices, cu_seqlens, _, _ = unpad_input(sequences.unsqueeze(-1), attention_mask)
    sequences = sequences.transpose(0, 1) # (1, total_seqs)
    packed_seq_lens = [cu_seqlens[i] - cu_seqlens[i-1] for i in range(1, len(cu_seqlens))]
    labels = index_first_axis(rearrange(labels.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1) # (1, total_seqs)
    position_ids = torch.clip(torch.cumsum(attention_mask, dim=-1) - 1, min=0, max=None)
    position_ids = index_first_axis(
        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1) # (1, total_seqs)
    ring_attn_pad_len = None
    if ring_attn_group is not None:
        ring_attn_size = dist.get_world_size(group=ring_attn_group)
        seqlen = sequences.shape[-1]
        # pad the sequences to divisible by ring_attn_size
        ring_attn_pad_len = (ring_attn_size - seqlen % ring_attn_size) % ring_attn_size
        pad_attn_mask_id = len(packed_seq_lens) + 1
        sequences = torch.nn.functional.pad(sequences, (0, ring_attn_pad_len), value=pad_attn_mask_id)
        attention_mask = torch.nn.functional.pad(attention_mask, (0, ring_attn_pad_len), value=pad_attn_mask_id)
        packed_seq_lens[-1] += ring_attn_pad_len
        labels = torch.nn.functional.pad(labels, (0, ring_attn_pad_len), value=0)
        # slice the sequences to current ring_attn_rank
        sequences, attention_mask, position_ids = convert_ring_attn_params(
            sequences, attention_mask, packed_seq_lens, ring_attn_group
        )
        start, end = get_slice_in_this_ring_attn_rank(labels.numel(), ring_attn_group)
        labels = labels[:, start:end]
    return sequences, attention_mask, position_ids, labels, ring_attn_pad_len, indices

def gather_and_pad_tensor(tensor, ring_attn_group, ring_attn_pad_len, indices, batch, seqlen):
    """
    Gather and pad tensor data (such as logits, log_probs, etc.).

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
        tensor = tensor[:, -ring_attn_pad_len:]
    tensor = pad_input(tensor.transpose(0, 1), indices, batch, seqlen).squeeze(-1)  # (batch, seqlen)
    return tensor
