"""
FSDP2 Utilities
===============

- moving_average_fsdp2: EMA update for FSDP2 models
- move_optimizer_state: Move optimizer state between devices
- clip_grad_norm_dtensor: Sharding-aware grad norm clipping for DTensor
- ensure_tied_word_embeddings: Keep tied embeddings stable
- reinit_rotary_embedding: Recompute rotary inv_freq after materialization
"""

import math
from typing import Callable

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor


def _group_parameters_by_sharding(
    parameters: list[torch.nn.Parameter],
) -> dict[tuple[object, object], list[torch.nn.Parameter]]:
    """Group parameters by DTensor sharding layout."""

    sharding_groups = {}
    for parameter in parameters:
        if isinstance(parameter, DTensor):
            key = (parameter.device_mesh, parameter.placements)
        else:
            key = ("regular", "regular")
        sharding_groups.setdefault(key, []).append(parameter)
    return sharding_groups


def _gather_norm_to_cpu(norm: torch.Tensor | DTensor) -> torch.Tensor:
    """Convert a possibly sharded norm tensor into a regular CPU scalar tensor."""

    if isinstance(norm, DTensor):
        if norm.to_local().device.type == "cpu" and torch.cuda.is_available():
            norm = norm.to(device=torch.device("cuda", torch.cuda.current_device()))
        norm = norm.full_tensor()
    return norm.detach().float().cpu()


def _compute_group_grad_norms(
    sharding_groups: dict[tuple[object, object], list[torch.nn.Parameter]],
    norm_type: float,
) -> list[torch.Tensor]:
    """Compute gradient norm for each sharding group, returned as CPU scalars."""

    group_grad_norms = []
    for group_parameters in sharding_groups.values():
        grads = [parameter.grad for parameter in group_parameters]
        norm = torch.nn.utils.get_total_norm(grads, norm_type, error_if_nonfinite=False)
        group_grad_norms.append(_gather_norm_to_cpu(norm))
    return group_grad_norms


def _reduce_group_grad_norms_to_total(group_grad_norms: list[torch.Tensor], norm_type: float) -> torch.Tensor:
    """Reduce per-group gradient norms into a single total norm."""

    if not group_grad_norms:
        return torch.tensor(0.0)
    if len(group_grad_norms) == 1:
        return group_grad_norms[0]
    if math.isinf(norm_type):
        return torch.stack(group_grad_norms).max()
    return sum(group_grad_norm**norm_type for group_grad_norm in group_grad_norms) ** (1.0 / norm_type)


@torch.no_grad()
def clip_grad_norm_dtensor(
    model: nn.Module,
    max_norm: float,
    norm_type: float = 2.0,
    total_norm: float | torch.Tensor | None = None,
) -> float:
    """Clip gradients for FSDP2 + TP models.

    With FSDP mesh = dp_cp, parameters are DTensors distributed across multiple
    dimensions. We:
    - Group parameters by (mesh, placements) since different sharding patterns
      cannot be mixed in a single clip_grad call.
    - Use `torch.nn.utils.get_total_norm`, which may return a DTensor with partial
      norm contributions; we materialize the global scalar via `full_tensor()`
      (this triggers the needed collectives for sharded grads).
    - Keep `total_norm` on CPU so `clip_grads_with_norm_` can safely apply the
      scale to both CPU and CUDA grads (it moves the scalar internally).

    Reference: Automodel's `_clip_grad_norm_impl` uses the same grouping idea.
    """
    parameters = [p for p in model.parameters() if p.grad is not None]
    if not parameters:
        return 0.0

    norm_type = float(norm_type)
    sharding_groups = _group_parameters_by_sharding(parameters)
    if total_norm is None:
        group_grad_norms = _compute_group_grad_norms(sharding_groups, norm_type)
        total_norm = _reduce_group_grad_norms_to_total(group_grad_norms, norm_type)
    else:
        total_norm = torch.as_tensor(total_norm, dtype=torch.float32).cpu()

    # If no clipping is needed, skip the (potentially large) in-place scaling pass.
    # total_norm is a CPU scalar tensor, so this check does not introduce GPU sync.
    if torch.isfinite(total_norm).item() and (total_norm <= max_norm).item():
        return total_norm.item()

    # Clip gradients for each sharding group separately
    for group_params in sharding_groups.values():
        torch.nn.utils.clip_grads_with_norm_(group_params, max_norm, total_norm)

    return total_norm.item()


@torch.no_grad()
def get_grad_norm_dtensor(
    model: nn.Module,
    norm_type: float = 2.0,
) -> float:
    """Return the current global gradient norm for a DTensor-sharded model."""

    parameters = [parameter for parameter in model.parameters() if parameter.grad is not None]
    if not parameters:
        return 0.0

    sharding_groups = _group_parameters_by_sharding(parameters)
    group_grad_norms = _compute_group_grad_norms(sharding_groups, float(norm_type))
    return _reduce_group_grad_norms_to_total(group_grad_norms, float(norm_type)).item()


def _dtensor_to_local(t: torch.Tensor) -> torch.Tensor:
    """Extract local shard from DTensor, or return tensor as-is.

    Under no_grad, DTensor.to_local() directly returns _local_tensor (no async).
    """
    return t.to_local() if isinstance(t, DTensor) else t


@torch.no_grad()
def moving_average_fsdp2(model: nn.Module, model_ema: nn.Module, unwrap_fn: Callable, beta: float = 0.992) -> None:
    """Update EMA model from a (typically) FSDP2-wrapped source model.

    Formula: ema = beta * ema + (1 - beta) * src

    Args:
        model: Source model (may be FSDP2-wrapped).
        model_ema: EMA model to update in-place.
        unwrap_fn: Function to unwrap model (e.g., get underlying module).
        beta: EMA decay factor in (0, 1). Default 0.992.

    Notes:
        - Assumes src/ema use the same sharding/layout; updates local shards in-place (no all-gather).
        - Supports cross-device/cross-dtype updates (e.g., CUDA actor -> CPU-offloaded EMA) via per-tensor cast/copy.
        - Buffers are copied (not EMA'd) to stay in sync with source.
        - beta must be in (0, 1); values outside this range are rejected.
    """
    if not 0.0 < beta < 1.0:
        raise ValueError(f"beta must be in (0, 1), got {beta}")

    src = unwrap_fn(model)
    ema = unwrap_fn(model_ema)

    lerp_weight = 1.0 - beta  # ema = ema + w * (src - ema) == (1-w)*ema + w*src

    def _update_tensor(name: str, dst_t: torch.Tensor, src_t: torch.Tensor, *, do_ema: bool) -> None:
        dst = _dtensor_to_local(dst_t)
        src_local = _dtensor_to_local(src_t)
        if dst.shape != src_local.shape:
            raise RuntimeError(
                f"EMA tensor shape mismatch for '{name}': ema={tuple(dst.shape)} src={tuple(src_local.shape)}"
            )
        if src_local.device != dst.device or src_local.dtype != dst.dtype:
            src_local = src_local.to(device=dst.device, dtype=dst.dtype, non_blocking=False)

        if do_ema:
            dst.lerp_(src_local, lerp_weight)
        else:
            dst.copy_(src_local)

    ema_params = dict(ema.named_parameters())
    for name, src_p in src.named_parameters():
        ema_p = ema_params.get(name)
        if ema_p is None:
            continue
        if not src_p.requires_grad:
            continue
        _update_tensor(name, ema_p, src_p, do_ema=True)

    ema_buffers = dict(ema.named_buffers())
    for name, src_b in src.named_buffers():
        ema_b = ema_buffers.get(name)
        if ema_b is None:
            continue
        _update_tensor(name, ema_b, src_b, do_ema=False)


@torch.no_grad()
def move_optimizer_state(optimizer: torch.optim.Optimizer, device: str | torch.device) -> None:
    """Move optimizer state tensors to specified device.

    Args:
        optimizer: PyTorch optimizer with state to move.
        device: Target device ("cpu", "cuda", or torch.device).

    Notes:
        Uses non_blocking transfer and synchronizes CUDA to ensure completion.
    """
    if not optimizer.state:
        return

    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state.get(param)
            if state is None:
                continue
            for key, val in state.items():
                if isinstance(val, torch.Tensor):
                    state[key] = val.to(device, non_blocking=True)

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def get_checkpoint_metadata(strategy) -> dict:
    """Build metadata for FSDP2 checkpoint (saved as fsdp2_runtime.json)."""
    return {
        "backend": "fsdp2",
        "world_size": strategy.world_size,
        "fsdp2_dp_size": strategy.fsdp2_dp_size,
        "fsdp2_cp_size": strategy.fsdp2_cp_size,
        "fsdp2_tp_size": strategy.fsdp2_tp_size,
        "param_dtype": strategy.param_dtype,
        "fsdp2_mesh_size": strategy.fsdp2_dp_size * strategy.fsdp2_cp_size,
    }


def ensure_tied_word_embeddings(model: nn.Module) -> bool:
    """Re-tie lm_head.weight to embed_tokens.weight if config requires."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    if not getattr(getattr(base, "config", None), "tie_word_embeddings", False):
        return False
    if hasattr(base, "tie_weights"):
        base.tie_weights()
        return True
    in_emb, out_emb = base.get_input_embeddings(), base.get_output_embeddings()
    if in_emb and out_emb:
        out_emb.weight = in_emb.weight
        return True
    return False


@torch.no_grad()
def reinit_rotary_embedding(backbone: nn.Module) -> None:
    """Recompute rotary inv_freq after meta-init materialization.

    HuggingFace registers inv_freq as a non-persistent buffer, so it's lost
    after to_empty() + checkpoint load and must be recomputed.
    """
    # HF places rotary_emb on LlamaModel (backbone.model) or directly on backbone.
    rotary_emb = getattr(getattr(backbone, "model", backbone), "rotary_emb", None)
    if rotary_emb is None or not all(hasattr(rotary_emb, name) for name in ("inv_freq", "config")):
        return

    device = rotary_emb.inv_freq.device

    if hasattr(rotary_emb, "rope_init_fn"):
        inv_freq, attention_scaling = rotary_emb.rope_init_fn(rotary_emb.config, device)
        rotary_emb.inv_freq.copy_(inv_freq.to(device=rotary_emb.inv_freq.device, dtype=rotary_emb.inv_freq.dtype))
        if hasattr(rotary_emb, "attention_scaling"):
            rotary_emb.attention_scaling = attention_scaling
        return

    try:
        fresh_rotary = type(rotary_emb)(rotary_emb.config, device=device)
    except TypeError:
        fresh_rotary = type(rotary_emb)(rotary_emb.config)

    for name in ("inv_freq", "original_inv_freq"):
        dst = getattr(rotary_emb, name, None)
        src = getattr(fresh_rotary, name, None)
        if torch.is_tensor(dst) and torch.is_tensor(src):
            dst.copy_(src.to(device=dst.device, dtype=dst.dtype))

    for attr in ("attention_scaling", "max_seq_len_cached", "original_max_seq_len", "rope_type"):
        if hasattr(fresh_rotary, attr):
            setattr(rotary_emb, attr, getattr(fresh_rotary, attr))
