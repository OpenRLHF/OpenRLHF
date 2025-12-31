"""
Checkpoint utilities for FSDP2.

Provides save/load for HuggingFace format and PyTorch distributed checkpoints.
"""

import gc
import os
import shutil
import warnings
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)
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
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> None:
    """Save model to HuggingFace format."""
    if is_rank_0:
        os.makedirs(output_dir, exist_ok=True)

    fsdp_model = unwrap_fn(model)
    is_peft = hasattr(fsdp_model, "peft_config")

    # Clean auto_map if needed
    config = getattr(fsdp_model, "config", None)
    if is_rank_0 and config and hasattr(config, "auto_map"):
        if isinstance(config.auto_map, dict) and None in config.auto_map:
            config.auto_map = {k: v for k, v in config.auto_map.items() if k is not None}

    # Get full state dict
    state = get_model_state_dict(
        fsdp_model,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True, ignore_frozen_params=is_peft),
    )

    if is_rank_0:
        fsdp_model.save_pretrained(output_dir, state_dict=state, **kwargs)
        _save_configs(fsdp_model, output_dir, tokenizer, metadata)

    _barrier()
    del state
    gc.collect()


def load_hf_model(
    model: nn.Module,
    path: str,
    unwrap_fn: UnwrapFn,
    map_location: str = "cpu",
    strict: bool = False,
    key_replace_fn: Optional[Callable] = None,
) -> None:
    """Load model weights from file."""
    state = torch.load(path, map_location=map_location)
    if key_replace_fn:
        state = key_replace_fn(state)

    wrapped = unwrap_fn(model)
    if not isinstance(wrapped, FSDPModule):
        raise RuntimeError("Model not wrapped. Call prepare() first.")

    set_model_state_dict(
        wrapped,
        state,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True, strict=strict),
    )


def _save_configs(model, output_dir: str, tokenizer, metadata: Optional[Dict] = None) -> None:
    """Save model configs, tokenizer, and metadata."""
    import json

    config = getattr(model, "config", None)
    if config:
        try:
            config.save_pretrained(output_dir)
        except Exception:
            try:
                config.to_json_file(os.path.join(output_dir, "config.json"))
            except Exception:
                pass

        if getattr(config, "auto_map", None):
            try:
                from transformers.dynamic_module_utils import custom_object_save

                custom_object_save(model, output_dir, config=config)
            except Exception:
                pass

    gen_config = getattr(model, "generation_config", None)
    if gen_config:
        try:
            gen_config.save_pretrained(output_dir)
        except Exception:
            pass

    if tokenizer:
        try:
            tokenizer.save_pretrained(output_dir)
        except Exception:
            pass

    if metadata:
        try:
            with open(os.path.join(output_dir, "fsdp2_runtime.json"), "w") as f:
                json.dump(metadata, f, indent=2, sort_keys=True)
        except Exception:
            pass


# =============================================================================
# Distributed Checkpoints
# =============================================================================


def save_distributed_checkpoint(
    model: nn.Module,
    save_dir: str,
    tag: str,
    unwrap_fn: UnwrapFn,
    is_rank_0: bool,
    optimizer: Optional[Optimizer] = None,
    scheduler=None,
    client_state: Optional[Dict[str, Any]] = None,
    max_num: int = 3,
    max_mem_gb: int = 1000,
    save_latest: bool = True,
) -> None:
    """Save FSDP2 distributed checkpoint."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict
    from torch.distributed.checkpoint.stateful import Stateful

    if not tag:
        raise ValueError("Checkpoint tag required")

    os.makedirs(save_dir, exist_ok=True)
    if is_rank_0:
        _cleanup_old_checkpoints(save_dir, max_num, max_mem_gb)
    _barrier()

    fsdp_model = unwrap_fn(model)
    ckpt_path = os.path.join(save_dir, tag)

    class AppState(Stateful):
        def __init__(self):
            self.model = fsdp_model
            self.optimizer = optimizer
            self.scheduler = scheduler
            self.client_state = dict(client_state or {})

        def state_dict(self):
            opts = [self.optimizer] if self.optimizer else []
            model_state, optim_state = get_state_dict(self.model, opts)
            state = {"model": model_state, "optimizers": optim_state, "client_state": self.client_state}
            if self.scheduler:
                state["scheduler"] = self.scheduler.state_dict()
            return state

        def load_state_dict(self, _):
            raise RuntimeError("Should not be called during save")

    if is_rank_0:
        os.makedirs(ckpt_path, exist_ok=True)
    _barrier()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=(FutureWarning, UserWarning))
        dcp.save({"app": AppState()}, checkpoint_id=ckpt_path)

    _barrier()

    if save_latest and is_rank_0:
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
) -> Tuple[str, Dict[str, Any]]:
    """Load FSDP2 distributed checkpoint."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_state_dict
    from torch.distributed.checkpoint.stateful import Stateful

    if not os.path.exists(load_dir):
        raise FileNotFoundError(f"[FSDP2] Checkpoint not found: {load_dir}")

    # Resolve tag
    resolved = tag
    if not resolved:
        latest_file = os.path.join(load_dir, "latest")
        if os.path.isfile(latest_file):
            with open(latest_file) as f:
                resolved = f.read().strip()
        else:
            subdirs = sorted(
                [d for d in os.listdir(load_dir) if os.path.isdir(os.path.join(load_dir, d))],
                key=lambda d: os.path.getmtime(os.path.join(load_dir, d)),
            )
            resolved = subdirs[-1] if subdirs else None

    if not resolved:
        raise FileNotFoundError("No checkpoint tag found")

    ckpt_path = os.path.join(load_dir, resolved)
    if not os.path.isdir(ckpt_path):
        raise FileNotFoundError(f"[FSDP2] Checkpoint path not found: {ckpt_path}")

    fsdp_model = unwrap_fn(model)
    load_opt = load_optimizer_states and optimizer and not load_module_only
    load_sched = load_lr_scheduler_states and scheduler and not load_module_only

    class AppState(Stateful):
        def __init__(self):
            self.model = fsdp_model
            self.optimizer = optimizer
            self.scheduler = scheduler
            self.client_state: Dict = {}

        def state_dict(self):
            raise RuntimeError("Should not be called during load")

        def load_state_dict(self, state):
            opts = [self.optimizer] if load_opt else []
            set_state_dict(
                self.model,
                opts,
                model_state_dict=state.get("model"),
                optim_state_dict=state.get("optimizers") if load_opt else {},
                options=StateDictOptions(strict=load_module_strict),
            )
            if load_sched and "scheduler" in state:
                self.scheduler.load_state_dict(state["scheduler"])
            self.client_state = state.get("client_state", {})

    app = AppState()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=(FutureWarning, UserWarning))
        dcp.load({"app": app}, checkpoint_id=ckpt_path)

    return resolved, app.client_state


# =============================================================================
# Helpers
# =============================================================================


def _barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def _cleanup_old_checkpoints(save_dir: str, max_num: int, max_mem_gb: int) -> None:
    """Remove old checkpoints based on count and size limits."""
    max_bytes = max_mem_gb * 1024**3

    def get_entries():
        entries = []
        for name in os.listdir(save_dir):
            path = os.path.join(save_dir, name)
            if os.path.isdir(path) and name != "latest":
                try:
                    entries.append((name, path, os.path.getmtime(path)))
                except OSError:
                    pass
        return sorted(entries, key=lambda x: x[2])

    def get_size(path):
        total = 0
        for root, _, files in os.walk(path):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
        return total

    entries = get_entries()

    # Remove by count
    while len(entries) > max_num:
        _, path, _ = entries.pop(0)
        shutil.rmtree(path, ignore_errors=True)

    # Remove by size
    total_size = sum(get_size(e[1]) for e in entries)
    while total_size > max_bytes and len(entries) > 1:
        _, path, _ = entries.pop(0)
        size = get_size(path)
        shutil.rmtree(path, ignore_errors=True)
        total_size -= size
