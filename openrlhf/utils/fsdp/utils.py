"""
FSDP2 Utilities
===============

- moving_average_fsdp: EMA update for FSDP models
- move_optimizer_state: Move optimizer state between devices
"""

import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# EMA (Exponential Moving Average)
# -----------------------------------------------------------------------------


def moving_average_fsdp(model: nn.Module, model_ema: nn.Module, unwrap_fn, beta: float = 0.992, device: str = "cpu"):
    """Update EMA model from FSDP-wrapped source model.

    Optimization strategies (ref: slime):
    1. If beta == 1.0: Use state_dict() copy (no communication needed)
    2. If EMA model is also FSDP-wrapped: Use state_dict() for sharded EMA
    3. Otherwise: Use full_tensor() with async redistribution to reduce blocking

    Args:
        model: Source model (FSDP-wrapped actor)
        model_ema: Target EMA model (may be Actor wrapper)
        unwrap_fn: Function to unwrap model (e.g., strategy._unwrap_model)
        beta: EMA decay factor
        device: Fallback device (used only if EMA param device detection fails)
    """
    from torch.distributed.fsdp import FSDPModule

    src = unwrap_fn(model)
    ema = unwrap_fn(model_ema)

    # Strategy 1: If beta == 1.0, just copy state_dict (no EMA, just copy)
    # This is the slime approach - most efficient, no all-gather needed
    if beta == 1.0:
        state = src.state_dict()
        ema.load_state_dict(state)
        return

    # Strategy 2: If EMA model is also FSDP-wrapped, operate on sharded params
    # This avoids full_tensor() all-gather by working with local shards only
    ema_is_fsdp = isinstance(ema, FSDPModule)
    if ema_is_fsdp:
        # Both models are FSDP-wrapped, use state_dict approach
        # Get sharded state dicts (no all-gather)
        src_state = src.state_dict()
        ema_state = ema.state_dict()

        with torch.no_grad():
            for name in src_state:
                if name in ema_state:
                    src_data = src_state[name]
                    ema_data = ema_state[name]
                    # EMA update on sharded data (no communication)
                    ema_data.mul_(beta).add_(src_data.to(ema_data.device), alpha=1 - beta)

        # Load updated state back
        ema.load_state_dict(ema_state)
        return

    # Strategy 3: Fallback - EMA model is not FSDP-wrapped
    # Use async redistribution to reduce blocking (ref: slime update_weight_utils.py)
    try:
        from torch.distributed.tensor import DTensor, Replicate
    except ImportError:
        DTensor = None

    ema_params = dict(ema.named_parameters())

    with torch.no_grad():
        for name, param in src.named_parameters():
            if param.requires_grad and name in ema_params:
                # Use async redistribution instead of blocking full_tensor() (ref: slime)
                if DTensor is not None and isinstance(param, DTensor):
                    # Async version: redistribute with async_op=True, then wait
                    data = param.redistribute(
                        placements=[Replicate()] * param.device_mesh.ndim,
                        async_op=True,
                    ).to_local()
                    # Wait for the async operation
                    if hasattr(data, "wait"):
                        data = data.wait()
                elif hasattr(param, "full_tensor"):
                    data = param.full_tensor()
                else:
                    data = param.data

                target_device = ema_params[name].device
                ema_params[name].data.mul_(beta).add_(data.to(target_device), alpha=1 - beta)


# -----------------------------------------------------------------------------
# Optimizer State Management
# -----------------------------------------------------------------------------


@torch.no_grad()
def move_optimizer_state(optimizer, device: torch.device):
    """Move all optimizer state tensors to specified device.

    This properly handles all tensor types in optimizer state,
    including nested lists/tuples (e.g., for AdamW's exp_avg, exp_avg_sq).
    """
    if not optimizer.state:
        return

    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state.get(param)
            if state is None:
                continue
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device, non_blocking=True)
                elif isinstance(v, (list, tuple)):
                    # Handle nested tensors in lists/tuples
                    state[k] = type(v)(t.to(device, non_blocking=True) if torch.is_tensor(t) else t for t in v)

    # Ensure all transfers complete before proceeding
    if device.type == "cuda":
        torch.cuda.synchronize()
    else:
        # For CPU offload, also sync to ensure memory is freed on GPU
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


# -----------------------------------------------------------------------------
# FSDP Model Offload/Reload (for vLLM sleep/wake_up compatibility)
# -----------------------------------------------------------------------------


@torch.no_grad()
def offload_fsdp_model_to_cpu(model: nn.Module, empty_cache: bool = True):
    """Offload FSDP2 model parameters to CPU.

    This is critical for vLLM sleep/wake_up mode to work with FSDP2.
    When vLLM wakes up, it needs to allocate CUDA memory for its model weights.
    If FSDP2 model is still on GPU, it causes CUDA OOM.

    Args:
        model: FSDP2-wrapped model
        empty_cache: Whether to empty CUDA cache after offloading
    """
    # Move model to CPU
    model.cpu()

    # CRITICAL: Synchronize to ensure all GPU->CPU transfers complete
    # before we try to free GPU memory
    torch.cuda.synchronize()

    if empty_cache:
        torch.cuda.empty_cache()


@torch.no_grad()
def load_fsdp_model_to_gpu(model: nn.Module, device_id: int = None):
    """Load FSDP2 model parameters back to GPU.

    Args:
        model: FSDP2-wrapped model (currently on CPU)
        device_id: Target CUDA device ID. If None, uses current device.
    """
    if device_id is None:
        device_id = torch.cuda.current_device()
    model.to(torch.device("cuda", device_id))


def get_runtime_metadata(strategy) -> dict:
    """Get runtime metadata for checkpoint saving."""
    return {
        "backend": "fsdp2",
        "world_size": getattr(strategy, "world_size", 1),
        "dp_size": getattr(strategy, "dp_size", 1),
        "ring_attn_size": getattr(strategy, "ring_attn_size", 1),
        "tp_size": getattr(strategy, "tp_size", 1),
        "precision": getattr(strategy, "precision", "bf16"),
    }
