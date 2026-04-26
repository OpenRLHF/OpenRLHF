"""vLLM weight refit helpers for the FSDP/Automodel backend.

Replaces the DS-era ``deepspeed.zero.GatheredParameters`` + ``GatherReplacedLayerParams``
context managers with a single ``gather_full_param`` call: under FSDP2, params are
``DTensor`` instances whose ``.full_tensor()`` materializes the unsharded tensor
across both FSDP shard and TP shard dims in one call.

The receiver-side plumbing in ``trainer/ray/ppo_actor.py`` (NCCL broadcast,
CUDA-IPC handle gather, vLLM ``update_weight`` RPC) is unchanged — only the
*gather* step swaps in.
"""

from typing import Tuple

import torch
from torch.distributed.tensor import DTensor


def gather_full_param(param: torch.nn.Parameter) -> Tuple[torch.Tensor, torch.Size]:
    """Materialize the full unsharded tensor for an FSDP/TP-sharded parameter.

    Returns ``(full_tensor, full_shape)`` where ``full_tensor`` is on the local
    device with all mesh dims gathered. For non-DTensor params (e.g., the value
    head we don't shard, or buffers), returns ``(param.data, param.shape)``.

    Caller invokes this on each rank; ``full_tensor`` is replicated. Memory cost
    is the size of the full tensor on every participating rank — acceptable for
    weight refit (one-shot per training step), but for very large models the
    Phase 4 async PPO path uses per-tensor streaming with a ping-pong buffer to
    bound peak memory.
    """
    if isinstance(param, DTensor):
        full = param.full_tensor()
        return full, full.shape
    return param.data, param.shape


def iter_named_full_params(model: torch.nn.Module, *, only_trainable: bool = False):
    """Iterate ``(name, full_tensor, full_shape)`` over a model's parameters,
    materializing DTensor shards on the fly. Caller is responsible for sending /
    broadcasting / IPC-handling each yielded tensor and releasing references
    promptly (drop refs before the next iteration to bound peak memory)."""
    for name, param in model.named_parameters():
        if only_trainable and not param.requires_grad:
            continue
        full, shape = gather_full_param(param)
        yield name, full, shape
