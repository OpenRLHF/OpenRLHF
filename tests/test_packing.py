"""Packing a padded batch into one sequence and back (openrlhf/models/ring_attn_utils.py)."""

import torch

from openrlhf.models.ring_attn_utils import gather_and_pad_tensor, unpad_and_slice_tensor


def test_unpad_then_pad_round_trip():
    sequences = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])

    packed, position_ids, rolled, pad_len, indices = unpad_and_slice_tensor(sequences, attention_mask, None)

    assert packed.tolist() == [[1, 2, 3, 4, 5]]
    assert position_ids.tolist() == [[0, 1, 2, 0, 1]]
    assert rolled.tolist() == [[2, 3, 0, 5, 0]]  # each token's successor within its own row
    assert pad_len == 0
    assert gather_and_pad_tensor(packed, None, pad_len, indices, batch=2, seqlen=4).tolist() == sequences.tolist()
