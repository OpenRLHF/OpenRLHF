"""Padded ↔ packed conversion for the FSDP/Automodel backend.

OpenRLHF's datasets emit `(B, S)` padded batches; Automodel/HF FlashAttention2
expects unpadded `(1, total_tokens)` packed input with `FlashAttentionKwargs`
(``cu_seq_lens_q`` / ``max_length_q`` per HF spec). This module provides the
runtime bridge.

Output is HF-canonical (``transformers.modeling_flash_attention_utils.FlashAttentionKwargs``):
both Automodel custom models and HF fallback models accept it via their
``**kwargs: Unpack[FlashAttentionKwargs]`` chain into the FA2 dispatcher.
"""

import torch
from torch.distributed.tensor import DTensor
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs


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


def pack_padded_batch(sequences: torch.Tensor, attention_mask: torch.Tensor):
    """Convert a padded `(B, S)` batch to packed `(1, total_real_tokens)` format.

    Returns:
        packed_input_ids: `(1, total_real_tokens)` — pad tokens removed
        position_ids:     `(1, total_real_tokens)` — resets at sequence boundaries
        rolled_input_ids: `(1, total_real_tokens)` — `torch.roll(input_ids, -1)` then unpadded
        indices:          flat indices into `(B*S,)` of real tokens (for `unpack_to_padded`)
        flash_attn_kwargs: HF ``FlashAttentionKwargs`` dict; spread into ``model.forward(**kwargs)``
    """
    from flash_attn.bert_padding import index_first_axis, rearrange, unpad_input

    rolled = torch.roll(sequences, shifts=-1, dims=1)
    packed_ids, indices, cu_seq_lens, max_length, _ = unpad_input(sequences.unsqueeze(-1), attention_mask)
    packed_ids = packed_ids.transpose(0, 1)  # (1, total)
    rolled_packed = index_first_axis(rearrange(rolled.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(
        0, 1
    )  # (1, total)

    # position_ids reset at seq boundaries (`[0,1,2, 0,1, 0,1,2,3]`).
    position_ids_full = torch.clip(torch.cumsum(attention_mask, dim=-1) - 1, min=0)
    position_ids = index_first_axis(
        rearrange(position_ids_full.unsqueeze(-1), "b s ... -> (b s) ..."), indices
    ).transpose(
        0, 1
    )  # (1, total)

    fa_kwargs: FlashAttentionKwargs = {
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
    from flash_attn.bert_padding import pad_input

    return pad_input(packed.transpose(0, 1), indices, batch, seqlen).squeeze(-1)
