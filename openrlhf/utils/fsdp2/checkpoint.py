"""
Checkpoint utilities for OpenRLHF FSDP2.

HF model export/import uses torch.distributed.checkpoint HF storage backends
(HuggingFaceStorageReader / HuggingFaceStorageWriter) with safetensors.
Training-state resume uses distributed DCP checkpoints.
"""

import json
import logging
import os
import re
import shutil
import warnings
from typing import Any, Callable, Dict, Optional

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.checkpoint._consolidate_hf_safetensors import consolidate_safetensors_files_on_every_rank
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.hf_storage import HuggingFaceStorageReader, HuggingFaceStorageWriter
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_state_dict,
    set_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim import Optimizer

from .utils import ensure_tied_word_embeddings

logger = logging.getLogger(__name__)

UnwrapFn = Callable[[nn.Module], nn.Module]


@torch.no_grad()
def _fixup_after_load(model: nn.Module) -> None:
    """Fixes after to_empty() + sharded load.

    - Re-tie embeddings if needed
    - Recompute rotary inv_freq (non-persistent buffer in some HF models)
    - Initialize reward-model mean/std buffers (persistent=False, not in ckpt)
    """
    backbone = model.get_base_model() if hasattr(model, "get_base_model") else model

    # Re-tie embeddings if required by config.
    ensure_tied_word_embeddings(backbone)

    # Rotary embedding inv_freq fix
    rotary_emb = None
    if hasattr(backbone, "model") and hasattr(backbone.model, "rotary_emb"):
        rotary_emb = backbone.model.rotary_emb
    elif hasattr(backbone, "rotary_emb"):
        rotary_emb = backbone.rotary_emb
    if rotary_emb is not None and hasattr(rotary_emb, "inv_freq") and hasattr(rotary_emb, "rope_init_fn") and hasattr(rotary_emb, "config"):
        device = rotary_emb.inv_freq.device if torch.is_tensor(rotary_emb.inv_freq) else torch.device("cpu")
        new_inv_freq, attention_scaling = rotary_emb.rope_init_fn(rotary_emb.config, device)
        if hasattr(rotary_emb, "attention_scaling"):
            rotary_emb.attention_scaling = attention_scaling
        rotary_emb.inv_freq.copy_(new_inv_freq)

    # Reward/Critic mean/std buffers.
    config = getattr(model, "config", None) or getattr(backbone, "config", None)
    for name in ("mean", "std"):
        if not hasattr(model, name):
            continue
        buffer = getattr(model, name)
        if not torch.is_tensor(buffer) or buffer.is_meta:
            continue
        if name == "mean":
            buffer.zero_()
        else:
            buffer.fill_(1.0)
        # If config carries normalization stats, prefer them.
        if config is not None and hasattr(config, name):
            buffer.fill_(float(getattr(config, name)))


def _compute_shard_mapping(
    state_dict: Dict[str, Any],
    max_shard_size_bytes: Optional[int],
) -> Optional[Dict[str, int]]:
    """Greedy mapping from fqn -> file index for HF sharded safetensors output.

    The mapping is consumed by `torch.distributed.checkpoint.hf_storage.HuggingFaceStorageWriter`
    when saving with `save_distributed=True` and consolidating shards into standard HF output.
    """
    if not max_shard_size_bytes or max_shard_size_bytes <= 0:
        return None

    mapping: Dict[str, int] = {}
    shard_index = 1
    shard_bytes = 0

    for key, tensor in state_dict.items():
        if not torch.is_tensor(tensor):
            continue
        tensor_bytes = tensor.numel() * tensor.element_size()
        if shard_bytes > 0 and shard_bytes + tensor_bytes > max_shard_size_bytes:
            shard_index += 1
            shard_bytes = 0
        mapping[key] = shard_index
        shard_bytes += tensor_bytes

    return mapping if mapping else None


def _cleanup_old_checkpoints(save_dir: str, max_num: int, *, tag: Optional[str] = None, is_rank_0: bool = True):
    """Write ``latest`` marker and remove old checkpoints under *save_dir*.

    Only rank-0 should perform filesystem mutations.  When *is_rank_0* is
    ``False`` the function is a no-op.  If *tag* is provided, a ``latest``
    file containing the tag name is written after cleanup.
    Set *max_num* to ``-1`` to disable deletion.
    """
    if not is_rank_0:
        return
    if max_num != -1 and max_num <= 0:
        raise ValueError(f"max_num must be -1 or a positive integer, got {max_num}")

    os.makedirs(save_dir, exist_ok=True)
    # Keep only canonical step directories to avoid deleting unrelated folders.
    step_dir_pattern = re.compile(r"^global_step_(\d+)$")
    checkpoints = []
    for name in os.listdir(save_dir):
        path = os.path.join(save_dir, name)
        if not os.path.isdir(path):
            continue
        match = step_dir_pattern.fullmatch(name)
        if match is None:
            continue
        checkpoints.append((int(match.group(1)), path))
    checkpoints.sort(key=lambda item: item[0])

    # Remove by count
    if max_num != -1:
        while len(checkpoints) > max_num:
            _, path = checkpoints.pop(0)
            shutil.rmtree(path)

    if tag is not None:
        with open(os.path.join(save_dir, "latest"), "w") as f:
            f.write(tag)


@torch.no_grad()
def _load_hf_checkpoint(
    model: nn.Module,
    model_name_or_path: str,
    *,
    device: torch.device | str = "cuda",
    process_group: Optional[dist.ProcessGroup] = None,
) -> nn.Module:
    """Load HF safetensors weights into an existing model via DCP.

    - ``process_group`` provided: distributed DCP load across FSDP2/TP ranks.
    - ``process_group`` omitted: single-process DCP load.

    The function materializes and updates ``model`` in-place, then returns the
    same object.
    """
    # Resolve HF repo id or local path into a local directory.
    hf_dir = os.path.expanduser(model_name_or_path)
    if not os.path.isdir(hf_dir):
        from huggingface_hub import snapshot_download
        hf_dir = snapshot_download(repo_id=model_name_or_path, repo_type="model")

    # DCP-based load (distributed or single-process).
    if process_group is None and not (dist.is_initialized() and dist.get_world_size() > 1):
        # Single-process: pick a real device
        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else "cpu"

    model.to_empty(device=device)

    if not any(f.endswith(".safetensors") for f in os.listdir(hf_dir)):
        raise FileNotFoundError(f"No .safetensors files found under: {hf_dir}")

    state_dict = model.state_dict()

    dcp.load(
        state_dict,
        storage_reader=HuggingFaceStorageReader(path=hf_dir),
        planner=DefaultLoadPlanner(allow_partial_load=True),
        process_group=process_group,
    )

    _fixup_after_load(model)

    if dist.is_initialized() and process_group is not None:
        dist.barrier(group=process_group)

    return model


def _save_hf_checkpoint(
    model: nn.Module,
    tokenizer,
    output_dir: str,
    is_rank_0: bool,
    *,
    process_group: Optional[dist.ProcessGroup] = None,
    max_shard_size_bytes: Optional[int] = None,
    thread_count: int = 8,
    thread_count_consolidation: int = 8,
    metadata: Optional[Dict] = None,
    save_dtype: Optional[torch.dtype] = None,
) -> None:
    """Save HF safetensors using DCP + HuggingFaceStorageWriter (distributed shards).

    Args:
        save_dtype: If provided, cast all parameters to this dtype before saving
            (e.g. ``torch.bfloat16``).  When ``None``, parameters are saved in
            their original (training) dtype.
    """
    if is_rank_0:
        os.makedirs(output_dir, exist_ok=True)

    # Sharded (distributed) state dict.
    model_state = get_model_state_dict(
        model, options=StateDictOptions(full_state_dict=False, cpu_offload=True)
    )

    # Cast to the desired save dtype (e.g. fp32 master weights → bf16 for deployment).
    if save_dtype is not None:
        model_state = {k: v.to(save_dtype) for k, v in model_state.items()}

    shard_mapping = _compute_shard_mapping(model_state, max_shard_size_bytes)

    staging_dir = os.path.join(output_dir, "sharded")
    if shard_mapping:
        # Multi-file output: save shards to sharded/, then consolidate in parallel across ranks.
        writer = HuggingFaceStorageWriter(
            path=staging_dir,
            save_distributed=True,
            fqn_to_index_mapping=shard_mapping,
            enable_consolidation=False,
            thread_count=thread_count,
        )
        dcp.save(model_state, storage_writer=writer, process_group=process_group)

        consolidate_safetensors_files_on_every_rank(
            input_dir=staging_dir,
            output_dir=output_dir,
            fqn_to_index_mapping=shard_mapping,
            num_threads=thread_count_consolidation,
            process_group=process_group,
        )
    else:
        # Single-file output: let HuggingFaceStorageWriter consolidate.
        writer = HuggingFaceStorageWriter(
            path=output_dir,
            save_distributed=True,
            enable_consolidation=True,
            thread_count=thread_count,
            thread_count_consolidation=thread_count_consolidation,
        )
        dcp.save(model_state, storage_writer=writer, process_group=process_group)

    if dist.is_initialized():
        dist.barrier(group=process_group)

    # Clean up sharded/ directory after consolidation.
    if is_rank_0 and os.path.exists(staging_dir):
        shutil.rmtree(staging_dir, ignore_errors=True)

    # Save config/tokenizer and runtime metadata (rank0 only).
    if is_rank_0:
        # Write HF-style safetensors index for sharded outputs.
        # The consolidate helpers in torch.distributed.checkpoint do not emit
        # `model.safetensors.index.json` today, but HuggingFace `from_pretrained`
        # expects it when multiple `model-0000x-of-0000y.safetensors` files exist.
        if shard_mapping:
            num_shards = max(shard_mapping.values()) if shard_mapping else 1
            index_path = os.path.join(output_dir, "model.safetensors.index.json")
            if not os.path.isfile(index_path):
                weight_map = {
                    key: f"model-{shard_idx:05d}-of-{num_shards:05d}.safetensors"
                    for key, shard_idx in shard_mapping.items()
                }
                total_size = 0
                for filename in set(weight_map.values()):
                    file_path = os.path.join(output_dir, filename)
                    if os.path.isfile(file_path):
                        try:
                            total_size += os.path.getsize(file_path)
                        except OSError:
                            pass
                with open(index_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {"metadata": {"total_size": total_size}, "weight_map": weight_map},
                        f,
                        indent=2,
                        sort_keys=True,
                    )

        # Try to save HF config from the model.
        config = getattr(model, "config", None)
        if config and hasattr(config, "auto_map") and isinstance(config.auto_map, dict):
            config.auto_map = {k: v for k, v in config.auto_map.items() if k is not None}
        if config and hasattr(config, "save_pretrained"):
            config.save_pretrained(output_dir)
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)
        if metadata:
            with open(os.path.join(output_dir, "fsdp2_runtime.json"), "w") as f:
                json.dump(metadata, f, indent=2, sort_keys=True)


def _save_dcp_checkpoint(
    model: nn.Module,
    save_dir: str,
    unwrap_fn: UnwrapFn,
    is_rank_0: bool,
    optimizer: Optional[Optimizer] = None,
    scheduler=None,
    client_state: Optional[Dict] = None,
    process_group: Optional[dist.ProcessGroup] = None,
):
    """Save FSDP2 distributed checkpoint.

    Each rank saves its own shard. Supports:
    - Model state (always saved)
    - Optimizer state (optional)
    - Scheduler state (optional)
    - Custom client state (e.g., consumed_samples)

    Args:
        save_dir: Exact DCP target directory, e.g.
            ``.../global_step_100/dcp_checkpoint/_actor``.
    """
    checkpoint_path = save_dir

    os.makedirs(save_dir, exist_ok=True)
    if dist.is_initialized():
        dist.barrier(group=process_group)

    unwrapped = unwrap_fn(model)

    # Stateful wrapper for DCP
    class AppState(Stateful):
        def state_dict(self):
            model_state, optim_state = get_state_dict(unwrapped, [optimizer] if optimizer else [])
            # NOTE: torch.distributed.checkpoint only persists tensor-like leaves
            # (Tensor/ShardedTensor/DTensor). Python objects (dict/int/float) in
            # the state dict are silently dropped in load. Persist those extras
            # (client_state, scheduler state) separately as a small rank0 file.
            return {"model": model_state, "optimizers": optim_state}

        def load_state_dict(self, _):
            raise RuntimeError("Use load_dcp_checkpoint instead")

    if is_rank_0:
        os.makedirs(checkpoint_path, exist_ok=True)
    if dist.is_initialized():
        dist.barrier(group=process_group)

    with warnings.catch_warnings():
        # `warnings.filterwarnings` expects `category` to be a Warning subclass,
        # not a tuple (Python 3.12 asserts this).
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        dcp.save({"app": AppState()}, checkpoint_id=checkpoint_path, process_group=process_group)

    if dist.is_initialized():
        dist.barrier(group=process_group)

    # Always write extra_state.pt and STABLE on rank0.
    if is_rank_0:
        extra_path = os.path.join(checkpoint_path, "extra_state.pt")
        torch.save(
            {
                "client_state": dict(client_state or {}),
                "scheduler": (scheduler.state_dict() if scheduler is not None else None),
            },
            extra_path,
        )
        # Mark checkpoint as complete to avoid picking up partial directories
        # when resolving checkpoints by mtime (prime-rl style).
        with open(os.path.join(checkpoint_path, "STABLE"), "w") as f:
            f.write("ok\n")

def _load_dcp_checkpoint(
    model: nn.Module,
    load_dir: str,
    unwrap_fn: UnwrapFn,
    optimizer: Optional[Optimizer] = None,
    scheduler=None,
    load_optimizer_states: bool = True,
    load_lr_scheduler_states: bool = True,
    load_module_strict: bool = True,
    load_module_only: bool = False,
    process_group: Optional[dist.ProcessGroup] = None,
) -> Dict[str, Any]:
    """Load FSDP2 distributed checkpoint.

    Args:
        load_dir: Exact DCP checkpoint directory.
        load_module_only: If True, skip optimizer/scheduler.
    """
    if not os.path.isdir(load_dir):
        raise FileNotFoundError(f"Checkpoint dir not found: {load_dir}")
    if not os.path.isfile(os.path.join(load_dir, ".metadata")):
        raise FileNotFoundError(f"Not a DCP checkpoint directory: {load_dir}")
    checkpoint_path = load_dir

    unwrapped = unwrap_fn(model)
    should_load_optimizer = load_optimizer_states and optimizer and not load_module_only
    should_load_scheduler = load_lr_scheduler_states and scheduler and not load_module_only

    # Stateful wrapper for loading
    class AppState(Stateful):
        def __init__(self):
            # DCP requires a template state_dict() describing what to load into.
            # In newer torch versions, dcp.load() will call state_dict() on the
            # provided Stateful object to build the load plan.
            model_state, optim_state = get_state_dict(unwrapped, [optimizer] if should_load_optimizer else [])
            self._template = {
                "model": model_state,
                "optimizers": optim_state,
            }

        def state_dict(self):
            return self._template

        def load_state_dict(self, state):
            set_state_dict(
                unwrapped,
                [optimizer] if should_load_optimizer else [],
                model_state_dict=state.get("model"),
                optim_state_dict=state.get("optimizers") if should_load_optimizer else {},
                options=StateDictOptions(strict=load_module_strict),
            )

    app = AppState()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        dcp.load({"app": app}, checkpoint_id=checkpoint_path, process_group=process_group)

    # Load non-tensor extras saved by rank0 (client_state + scheduler state).
    extra_data = {}
    extra_path = os.path.join(checkpoint_path, "extra_state.pt")
    if dist.is_initialized():
        if dist.get_rank() == 0 and os.path.isfile(extra_path):
            extra_data = torch.load(extra_path, map_location="cpu")
    elif os.path.isfile(extra_path):
        extra_data = torch.load(extra_path, map_location="cpu")
    if dist.is_initialized():
        broadcast_list = [extra_data]
        dist.broadcast_object_list(broadcast_list, src=0, group=process_group)
        extra_data = broadcast_list[0] or {}

    client_state = extra_data.get("client_state", {}) if isinstance(extra_data, dict) else {}
    if should_load_scheduler and scheduler is not None:
        scheduler_state = extra_data.get("scheduler", None) if isinstance(extra_data, dict) else None
        if scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)

    return client_state
