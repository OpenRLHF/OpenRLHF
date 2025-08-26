from __future__ import annotations

import torch

from .ring_attn_utils import (
    unpad_and_slice_tensor as ring_unpad_and_slice_tensor,
    gather_and_pad_tensor as ring_gather_and_pad_tensor,
)
from .star_attn_utils import (
    unpad_and_slice_tensor as star_unpad_and_slice_tensor,
    gather_and_pad_tensor as star_gather_and_pad_tensor,
    get_star_attn_group,
)


def unpad_and_slice_auto(sequences: torch.Tensor, attention_mask: torch.Tensor, ring_attn_group):
    """
    Route to Ring or Star unpad/slice based on provided group.

    Returns:
        sequences, position_ids, rolled_sequences, attn_pad_len, indices, attn_kind, attn_group
    """
    if ring_attn_group is not None:
        sequences, position_ids, rolled_sequences, attn_pad_len, indices = ring_unpad_and_slice_tensor(
            sequences, attention_mask, ring_attn_group
        )
        attn_kind = "ring"
        attn_group = ring_attn_group
    else:
        attn_group = get_star_attn_group()
        sequences, position_ids, rolled_sequences, attn_pad_len, indices = star_unpad_and_slice_tensor(
            sequences, attention_mask, attn_group
        )
        attn_kind = "star"
    return sequences, position_ids, rolled_sequences, attn_pad_len, indices, attn_kind, attn_group


def gather_and_pad_auto(tensor: torch.Tensor, attn_kind: str, attn_group, attn_pad_len: int, indices, batch: int, seqlen: int):
    if attn_kind == "star":
        return star_gather_and_pad_tensor(tensor, attn_group, attn_pad_len, indices, batch, seqlen)
    return ring_gather_and_pad_tensor(tensor, attn_group, attn_pad_len, indices, batch, seqlen)


