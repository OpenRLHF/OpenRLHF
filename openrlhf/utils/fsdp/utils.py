"""
FSDP2 Utilities
===============

- clip_grad_norm_dtensor: DTensor-safe gradient clipping
- moving_average_fsdp: EMA update for FSDP models
- move_optimizer_state: Move optimizer state between devices
"""

import math
import torch
import torch.distributed as dist
import torch.nn as nn


# -----------------------------------------------------------------------------
# Gradient Clipping (DTensor-safe)
# -----------------------------------------------------------------------------


def clip_grad_norm_dtensor(model: nn.Module, max_norm: float, norm_type: float = 2.0) -> float:
    """Clip gradients, handling DTensor from FSDP/TP correctly.

    Unlike torch.nn.utils.clip_grad_norm_, this handles:
    - DTensor gradients (from FSDP2/TP) that need full_tensor() reduction
    - Mixed scenarios with both DTensor and regular tensors
    """
    if max_norm <= 0:
        return 0.0

    params = [p for p in model.parameters() if p.grad is not None]
    if not params:
        return 0.0

    # Non-distributed: use standard clipping
    if not dist.is_initialized():
        return float(torch.nn.utils.clip_grad_norm_(params, max_norm, norm_type))

    grads = [p.grad for p in params]
    device = grads[0].device

    # Try standard API first (works for uniform DTensor)
    try:
        total_norm = torch.nn.utils.get_total_norm(grads, norm_type, False, False)
        total_norm = _to_full_tensor(total_norm)
    except Exception:
        # Fallback for mixed DTensor/regular tensor scenarios
        total_norm = _compute_mixed_norm(grads, device, norm_type)

    total_norm = total_norm.to(device, non_blocking=True)
    torch.nn.utils.clip_grads_with_norm_(params, max_norm, total_norm, False)
    return float(total_norm)


def _to_full_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Convert DTensor to full tensor if needed."""
    try:
        from torch.distributed.tensor import DTensor

        if isinstance(tensor, DTensor):
            return tensor.full_tensor()
    except ImportError:
        pass
    return tensor


def _compute_mixed_norm(grads, device, norm_type):
    """Compute norm for mixed DTensor/regular tensor gradients."""
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        DTensor = None

    # Group by mesh for correct reduction
    mesh_groups, local = {}, []
    for g in grads:
        if DTensor and isinstance(g, DTensor):
            key = id(g.device_mesh)
            mesh_groups.setdefault(key, []).append(g)
        else:
            local.append(g)

    is_inf = math.isinf(norm_type)
    total = torch.zeros((), device=device)

    for group in list(mesh_groups.values()) + ([local] if local else []):
        norm = torch.nn.utils.get_total_norm(group, norm_type, False, False)
        norm = _to_full_tensor(norm).to(device)
        total = torch.maximum(total, norm) if is_inf else total + norm.pow(norm_type)

    return total if is_inf else total.pow(1.0 / norm_type)


# -----------------------------------------------------------------------------
# EMA (Exponential Moving Average)
# -----------------------------------------------------------------------------


def moving_average_fsdp(model: nn.Module, model_ema: nn.Module, unwrap_fn, beta: float = 0.992, device: str = "cpu"):
    """Update EMA model from FSDP-wrapped source model.

    Efficient implementation that:
    - Uses full_tensor() to handle sharded params
    - Only updates trainable params that exist in both models
    - Handles device placement correctly (uses EMA param's actual device)

    Args:
        model: Source model (FSDP-wrapped actor)
        model_ema: Target EMA model (may be Actor wrapper)
        unwrap_fn: Function to unwrap model (e.g., strategy._unwrap_model)
        beta: EMA decay factor
        device: Fallback device (used only if EMA param device detection fails)
    """
    src = unwrap_fn(model)
    ema = unwrap_fn(model_ema)
    ema_params = dict(ema.named_parameters())

    with torch.no_grad():
        for name, param in src.named_parameters():
            if param.requires_grad and name in ema_params:
                data = param.full_tensor() if hasattr(param, "full_tensor") else param.data
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
