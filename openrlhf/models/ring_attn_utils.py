import torch
import torch.distributed as dist
import torch.nn.functional as F


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


def convert_ring_attn_params(sequences, attention_mask, packed_seq_lens, ring_attn_group):
    # each rank within the ring group will process sequences[start:end]
    ring_attn_rank = dist.get_rank(group=ring_attn_group)
    ring_attn_size = dist.get_world_size(group=ring_attn_group)
    total_seq_len = sequences.numel()
    local_seq_len = total_seq_len // ring_attn_size
    start, end = ring_attn_rank * local_seq_len, (ring_attn_rank + 1) * local_seq_len
    sequences = sequences[:, start:end]
    attention_mask = attention_mask[:, start:end]
    position_ids = reset_ring_attn_position_ids(start, end, packed_seq_lens)
    update_ring_attn_params(packed_seq_lens, total_seq_len)
    return sequences, attention_mask, position_ids


def pad_sequences(sequences, attention_mask, num_actions, packed_seq_lens, ring_attn_group, pad_token_id=0):
    # Pads the input sequences and attention mask to ensure that their lengths are multiples of the ring attention size.
    ring_attn_size = dist.get_world_size(group=ring_attn_group)
    if isinstance(sequences, torch.Tensor):
        seqlen = sequences.shape[-1]
        pad_len = (ring_attn_size - seqlen % ring_attn_size) % ring_attn_size
        padded = torch.tensor([pad_token_id] * pad_len, device=sequences.device, dtype=sequences.dtype).unsqueeze(0)
        sequences = torch.cat([sequences, padded], dim=1)
        attention_mask = torch.cat(
            [attention_mask, (len(sequences) + 1) * torch.ones(1, pad_len, device="cuda", dtype=torch.float)], dim=-1
        )
    elif isinstance(sequences, list):
        seqlen = len(sequences)
        pad_len = (ring_attn_size - seqlen % ring_attn_size) % ring_attn_size
        sequences += [pad_token_id] * pad_len
        attention_mask += [len(packed_seq_lens) + 1] * pad_len
    else:
        raise "sequences is not available type"
    num_actions[-1] += pad_len
    packed_seq_lens[-1] += pad_len
    return pad_len, sequences, attention_mask, num_actions, packed_seq_lens


def unpad_sequences(
    pad_len,
    sequences,
    attention_mask,
    num_actions,
    packed_seq_lens,
    ring_attn_group,
    action_log_probs=None,
    values=None,
    kl=None,
):
    # Removes the padding from the input sequences, attention mask, and other optional tensors after padding.
    if pad_len > 0:
        sequences = sequences[:, :-pad_len]
        attention_mask = attention_mask[:, :-pad_len]
        num_actions[-1] -= pad_len
        packed_seq_lens[-1] -= pad_len
        if action_log_probs is not None:
            action_log_probs = action_log_probs[:, :-pad_len]
        if values is not None:
            values = values[:, :-pad_len]
        if kl is not None:
            kl = kl[:, :-pad_len]
    return sequences, attention_mask, num_actions, packed_seq_lens, action_log_probs, values, kl
