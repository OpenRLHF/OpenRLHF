"""
Checkpoint Utilities for FSDP2
==============================

Two checkpoint formats:
1. HuggingFace format: save_hf_model/load_hf_model
   - Standard HF format, compatible with transformers
   - Full state dict gathered to rank 0

2. Distributed checkpoints: save_distributed_checkpoint/load_distributed_checkpoint
   - PyTorch DCP format, each rank saves its shard
   - Supports resuming training with optimizer/scheduler state
"""

import json
import os
import shutil
import warnings
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_state_dict,
    set_model_state_dict,
    set_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.fsdp import FSDPModule
from torch.optim import Optimizer

UnwrapFn = Callable[[nn.Module], nn.Module]


# =============================================================================
# HuggingFace Format
# =============================================================================


def save_hf_model(
    model: nn.Module,
    tokenizer,
    output_dir: str,
    is_rank_0: bool,
    unwrap_fn: UnwrapFn,
    metadata: Optional[Dict] = None,
    **kwargs,
):
    """Save model to HuggingFace format.

    Gathers full state dict to rank 0 and saves using model.save_pretrained().
    For PEFT models, extracts adapter weights using get_peft_model_state_dict.
    Also saves tokenizer, config, and optional runtime metadata.
    """
    if is_rank_0:
        os.makedirs(output_dir, exist_ok=True)

    fsdp_model = unwrap_fn(model)
    is_peft = hasattr(fsdp_model, "peft_config")

    # Get full state dict (gathered to rank 0)
    state = get_model_state_dict(fsdp_model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))

    if is_rank_0:
        # Clean up auto_map if corrupted
        cfg = getattr(fsdp_model, "config", None)
        if cfg and hasattr(cfg, "auto_map") and isinstance(cfg.auto_map, dict):
            cfg.auto_map = {k: v for k, v in cfg.auto_map.items() if k is not None}

        if is_peft:
            # PEFT model: extract adapter weights using get_peft_model_state_dict
            # Keep adapter artifacts consistent with other checkpoint writers
            from peft.utils.save_and_load import get_peft_model_state_dict

            # Save adapter config and weights (PEFT internally extracts adapter weights from state)
            fsdp_model.save_pretrained(output_dir, state_dict=state, **kwargs)
            # Extract adapter weights and save as .bin format for compatibility
            adapter_state = get_peft_model_state_dict(fsdp_model, state_dict=state)
            torch.save(adapter_state, os.path.join(output_dir, "adapter_model.bin"))
            # Remove safetensors file if exists (avoid conflicts)
            safetensors_path = os.path.join(output_dir, "adapter_model.safetensors")
            if os.path.exists(safetensors_path):
                os.remove(safetensors_path)
        else:
            # Regular model: save directly
            fsdp_model.save_pretrained(output_dir, state_dict=state, **kwargs)

        if tokenizer:
            tokenizer.save_pretrained(output_dir)
        if metadata:
            with open(os.path.join(output_dir, "fsdp2_runtime.json"), "w") as f:
                json.dump(metadata, f, indent=2, sort_keys=True)

    dist.barrier()
    del state


def load_hf_model(
    model: nn.Module,
    path: str,
    unwrap_fn: UnwrapFn,
    map_location: str = "cpu",
    strict: bool = False,
    key_replace_fn: Optional[Callable] = None,
):
    """Load model weights from a saved checkpoint file.

    Args:
        path: Path to saved weights (e.g., pytorch_model.bin)
        key_replace_fn: Optional function to transform state dict keys
    """
    # Avoid loading full weights on every rank in FSDP2; let rank 0 load
    # and broadcast to others (verl-style).
    if dist.is_initialized() and dist.get_rank() != 0:
        state = {}
    else:
        state = torch.load(path, map_location=map_location)
        if key_replace_fn:
            state = key_replace_fn(state)

    wrapped = unwrap_fn(model)
    if not isinstance(wrapped, FSDPModule):
        raise RuntimeError("Model not FSDP-wrapped. Call strategy.prepare() first.")

    set_model_state_dict(
        wrapped,
        state,
        options=StateDictOptions(
            full_state_dict=True,
            cpu_offload=True,
            strict=strict,
            broadcast_from_rank0=dist.is_initialized(),
        ),
    )


# =============================================================================
# Distributed Checkpoints (PyTorch DCP)
# =============================================================================


def save_distributed_checkpoint(
    model: nn.Module,
    save_dir: str,
    tag: str,
    unwrap_fn: UnwrapFn,
    is_rank_0: bool,
    optimizer: Optional[Optimizer] = None,
    scheduler=None,
    client_state: Optional[Dict] = None,
    max_num: int = 3,
    max_mem: int = 1000,
    save_latest: bool = True,
    process_group: Optional[dist.ProcessGroup] = None,
):
    """Save FSDP2 distributed checkpoint.

    Each rank saves its own shard. Supports:
    - Model state (always saved)
    - Optimizer state (optional)
    - Scheduler state (optional)
    - Custom client state (e.g., consumed_samples)

    Args:
        tag: Checkpoint name (e.g., "step_1000")
        max_num: Maximum number of checkpoints to keep
        max_mem: Maximum total checkpoint size in GB
    """
    if not tag:
        raise ValueError("Checkpoint tag required")

    os.makedirs(save_dir, exist_ok=True)
    if is_rank_0:
        _cleanup_old_checkpoints(save_dir, max_num, max_mem)
    dist.barrier(group=process_group) if process_group is not None else dist.barrier()

    fsdp_model = unwrap_fn(model)
    ckpt_path = os.path.join(save_dir, tag)

    # Stateful wrapper for DCP
    class AppState(Stateful):
        def state_dict(self):
            model_state, optim_state = get_state_dict(fsdp_model, [optimizer] if optimizer else [])
            # NOTE: torch.distributed.checkpoint only persists tensor-like leaves
            # (Tensor/ShardedTensor/DTensor). Python objects (dict/int/float) in
            # the state dict are silently dropped in load. Persist those extras
            # (client_state, scheduler state) separately as a small rank0 file.
            return {"model": model_state, "optimizers": optim_state}

        def load_state_dict(self, _):
            raise RuntimeError("Use load_distributed_checkpoint instead")

    if is_rank_0:
        os.makedirs(ckpt_path, exist_ok=True)
    dist.barrier(group=process_group) if process_group is not None else dist.barrier()

    with warnings.catch_warnings():
        # `warnings.filterwarnings` expects `category` to be a Warning subclass,
        # not a tuple (Python 3.12 asserts this).
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        dcp.save({"app": AppState()}, checkpoint_id=ckpt_path, process_group=process_group)

    dist.barrier(group=process_group) if process_group is not None else dist.barrier()

    # Update 'latest' marker
    if save_latest and is_rank_0:
        # Save non-tensor extras (client_state, scheduler) on rank0.
        extra_path = os.path.join(ckpt_path, "extra_state.pt")
        torch.save(
            {
                "client_state": dict(client_state or {}),
                "scheduler": (scheduler.state_dict() if scheduler is not None else None),
            },
            extra_path,
        )
        # Mark checkpoint as complete to avoid picking up partial directories
        # when resolving checkpoints by mtime (prime-rl style).
        with open(os.path.join(ckpt_path, "STABLE"), "w") as f:
            f.write("ok\n")
        with open(os.path.join(save_dir, "latest"), "w") as f:
            f.write(tag)


def load_distributed_checkpoint(
    model: nn.Module,
    load_dir: str,
    unwrap_fn: UnwrapFn,
    tag: Optional[str] = None,
    optimizer: Optional[Optimizer] = None,
    scheduler=None,
    load_optimizer_states: bool = True,
    load_lr_scheduler_states: bool = True,
    load_module_strict: bool = True,
    load_module_only: bool = False,
    process_group: Optional[dist.ProcessGroup] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Load FSDP2 distributed checkpoint.

    Args:
        tag: Checkpoint tag (reads 'latest' file if None)
        load_module_only: If True, skip optimizer/scheduler

    Returns:
        (tag, client_state) tuple
    """
    if not os.path.exists(load_dir):
        raise FileNotFoundError(f"Checkpoint dir not found: {load_dir}")

    # Resolve tag from 'latest' file or find most recent
    resolved = tag
    if not resolved:
        latest_file = os.path.join(load_dir, "latest")
        if os.path.isfile(latest_file):
            with open(latest_file) as f:
                resolved = f.read().strip()
        else:
            subdirs = sorted(
                [d for d in os.listdir(load_dir) if os.path.isdir(os.path.join(load_dir, d)) and d != "latest"],
                key=lambda d: os.path.getmtime(os.path.join(load_dir, d)),
            )
            # Prefer completed checkpoints if STABLE markers exist.
            stable_subdirs = [d for d in subdirs if os.path.isfile(os.path.join(load_dir, d, "STABLE"))]
            if stable_subdirs:
                subdirs = stable_subdirs
            resolved = subdirs[-1] if subdirs else None

    if not resolved:
        raise FileNotFoundError("No checkpoint tag found")

    ckpt_path = os.path.join(load_dir, resolved)
    if not os.path.isdir(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    fsdp_model = unwrap_fn(model)
    load_opt = load_optimizer_states and optimizer and not load_module_only
    load_sched = load_lr_scheduler_states and scheduler and not load_module_only

    # Stateful wrapper for loading
    class AppState(Stateful):
        def __init__(self):
            # DCP requires a template state_dict() describing what to load into.
            # In newer torch versions, dcp.load() will call state_dict() on the
            # provided Stateful object to build the load plan.
            model_state, optim_state = get_state_dict(fsdp_model, [optimizer] if load_opt else [])
            self._template = {
                "model": model_state,
                "optimizers": optim_state,
            }

        def state_dict(self):
            return self._template

        def load_state_dict(self, state):
            set_state_dict(
                fsdp_model,
                [optimizer] if load_opt else [],
                model_state_dict=state.get("model"),
                optim_state_dict=state.get("optimizers") if load_opt else {},
                options=StateDictOptions(strict=load_module_strict),
            )

    app = AppState()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        dcp.load({"app": app}, checkpoint_id=ckpt_path, process_group=process_group)

    # Load non-tensor extras saved by rank0 (client_state + scheduler state).
    extra_state = {}
    extra_path = os.path.join(ckpt_path, "extra_state.pt")
    if dist.is_initialized() and dist.get_rank() == 0 and os.path.isfile(extra_path):
        extra_state = torch.load(extra_path, map_location="cpu")
    if dist.is_initialized():
        obj_list = [extra_state]
        dist.broadcast_object_list(obj_list, src=0, group=process_group)
        extra_state = obj_list[0] or {}

    client_state = extra_state.get("client_state", {}) if isinstance(extra_state, dict) else {}
    if load_sched and scheduler is not None:
        sched_state = extra_state.get("scheduler", None) if isinstance(extra_state, dict) else None
        if sched_state is not None:
            scheduler.load_state_dict(sched_state)

    return resolved, client_state


def _cleanup_old_checkpoints(save_dir: str, max_num: int, max_mem: int):
    """Remove old checkpoints to stay within count and size limits."""
    max_bytes = max_mem * 1024**3

    # Get checkpoint entries sorted by modification time
    entries = []
    for name in os.listdir(save_dir):
        path = os.path.join(save_dir, name)
        if os.path.isdir(path) and name != "latest":
            try:
                entries.append((name, path, os.path.getmtime(path)))
            except OSError:
                pass
    entries.sort(key=lambda x: x[2])

    def get_size(path):
        return sum(os.path.getsize(os.path.join(r, f)) for r, _, files in os.walk(path) for f in files)

    # Remove by count
    while len(entries) > max_num:
        shutil.rmtree(entries.pop(0)[1], ignore_errors=True)

    # Remove by size
    total = sum(get_size(e[1]) for e in entries)
    while total > max_bytes and len(entries) > 1:
        path = entries.pop(0)[1]
        total -= get_size(path)
        shutil.rmtree(path, ignore_errors=True)
