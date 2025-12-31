"""FSDP2 utilities: gradient clipping, EMA, optimizer state management."""

import math
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn


# =============================================================================
# Actor Unwrapping
# =============================================================================


def unwrap_actor(model: nn.Module) -> nn.Module:
    """Unwrap Actor wrapper to get inner model."""
    try:
        from openrlhf.models import Actor

        if isinstance(model, Actor):
            return unwrap_actor(model.model)
    except ImportError:
        pass
    return model


# =============================================================================
# Gradient Clipping
# =============================================================================


def clip_grad_norm_dtensor(
    model: nn.Module,
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: Optional[bool] = None,
) -> float:
    """DTensor-safe gradient clipping."""
    if max_norm <= 0:
        return 0.0

    params = [p for p in model.parameters() if p.grad is not None]
    if not params:
        return 0.0

    if not dist.is_initialized():
        return float(torch.nn.utils.clip_grad_norm_(params, max_norm, norm_type, error_if_nonfinite, foreach))

    grads = [p.grad for p in params]
    device = grads[0].device if grads else torch.device("cuda")

    try:
        total_norm = torch.nn.utils.get_total_norm(grads, norm_type, error_if_nonfinite, foreach or False)
        total_norm = _reduce_dtensor(total_norm)
    except Exception:
        total_norm = _compute_mixed_norm(grads, device, norm_type, error_if_nonfinite, foreach)

    total_norm = total_norm.to(device, non_blocking=True)
    torch.nn.utils.clip_grads_with_norm_(params, max_norm, total_norm, foreach or False)
    return float(total_norm.item() if hasattr(total_norm, "item") else total_norm)


def _reduce_dtensor(tensor: torch.Tensor) -> torch.Tensor:
    """Convert DTensor to regular tensor."""
    try:
        from torch.distributed.tensor import DTensor

        if isinstance(tensor, DTensor):
            return tensor.full_tensor()
    except ImportError:
        pass
    return tensor


def _compute_mixed_norm(grads, device, norm_type, error_if_nonfinite, foreach):
    """Compute gradient norm for mixed DTensor scenarios."""
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        DTensor = None

    mesh_groups: Dict[Any, list] = {}
    local_grads = []

    for grad in grads:
        if DTensor and isinstance(grad, DTensor):
            placements = getattr(grad, "placements", None)
            key = (
                id(grad.device_mesh),
                tuple((p.__class__.__name__, getattr(p, "dim", None)) for p in placements) if placements else None,
            )
            mesh_groups.setdefault(key, []).append(grad)
        else:
            local_grads.append(grad)

    is_inf = math.isinf(norm_type)
    total = torch.zeros((), device=device)
    total_pow = torch.zeros((), device=device)

    def add_norm(norm):
        nonlocal total, total_pow
        norm = _reduce_dtensor(norm).to(device, non_blocking=True)
        if is_inf:
            total = torch.maximum(total, norm)
        else:
            total_pow += norm.pow(norm_type)

    for group in mesh_groups.values():
        add_norm(torch.nn.utils.get_total_norm(group, norm_type, error_if_nonfinite, bool(foreach)))

    if local_grads:
        add_norm(torch.nn.utils.get_total_norm(local_grads, norm_type, error_if_nonfinite, bool(foreach)))

    return total if is_inf else total_pow.pow(1.0 / norm_type)


# =============================================================================
# EMA
# =============================================================================


def moving_average_fsdp(
    model: nn.Module,
    model_ema: nn.Module,
    beta: float = 0.992,
    device: str = "cpu",
) -> None:
    """Update EMA model efficiently."""
    wrapped = unwrap_actor(model)
    ema_params = dict(model_ema.named_parameters())

    with torch.no_grad():
        for name, param in wrapped.named_parameters():
            if not param.requires_grad or name not in ema_params:
                continue
            data = param.full_tensor() if hasattr(param, "full_tensor") else param.data
            ema_params[name].data.mul_(beta).add_(data.to(device), alpha=1 - beta)


# =============================================================================
# Optimizer State
# =============================================================================


def move_optimizer_state(optimizer, device: torch.device) -> None:
    """Move optimizer state to specified device."""
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device, non_blocking=True)


def get_runtime_metadata(strategy) -> Dict[str, Any]:
    """Get runtime metadata for checkpointing."""
    return {
        "backend": "fsdp2",
        "world_size": getattr(strategy, "world_size", 1),
        "dp_size": getattr(strategy, "dp_size", 1),
        "ring_attn_size": getattr(strategy, "ring_attn_size", 1),
        "tp_size": getattr(strategy, "tp_size", 1),
        "precision": getattr(strategy, "precision", "bf16"),
        "fsdp2_offload": getattr(strategy, "fsdp2_offload", "none"),
    }
