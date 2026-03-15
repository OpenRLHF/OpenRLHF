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

from .utils import ensure_tied_word_embeddings, reinit_rotary_embedding

logger = logging.getLogger(__name__)

UnwrapFn = Callable[[nn.Module], nn.Module]


@torch.no_grad()
def _fixup_after_load(model: nn.Module) -> None:
    """Reinitialize non-persistent state after to_empty() + DCP load.

    Delegates to reusable utilities in utils.py and model-level hooks:
    1. Re-tie word embeddings (backbone)
    2. Recompute rotary inv_freq (backbone)
    3. Reset model-specific buffers (e.g. reward normalization stats)
    """
    backbone = model.get_base_model() if hasattr(model, "get_base_model") else model
    ensure_tied_word_embeddings(backbone)
    reinit_rotary_embedding(backbone)
    if hasattr(model, "reset_buffers"):
        model.reset_buffers()


def _read_safetensors_keys(hf_dir: str) -> set[str]:
    """Return tensor keys stored in an HF safetensors checkpoint directory."""
    index_path = os.path.join(hf_dir, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as f:
            index_data = json.load(f)
        return set(index_data.get("weight_map", {}))

    from safetensors import safe_open

    checkpoint_keys: set[str] = set()
    for filename in sorted(name for name in os.listdir(hf_dir) if name.endswith(".safetensors")):
        file_path = os.path.join(hf_dir, filename)
        with safe_open(file_path, framework="pt", device="cpu") as f:
            checkpoint_keys.update(f.keys())
    return checkpoint_keys


def _get_required_missing_hf_keys(model: nn.Module, checkpoint_keys: set[str], state_dict_keys: set[str]) -> list[str]:
    """Return model keys absent from the checkpoint that callers must handle.

    Filters out tied-weight aliases (via ``all_tied_weights_keys``) and
    model-declared ignorable patterns (``_keys_to_ignore_on_load_missing``),
    consistent with how ``transformers.PreTrainedModel.from_pretrained`` works.
    """
    missing = state_dict_keys - checkpoint_keys

    # Remove tied-weight aliases.  all_tied_weights_keys is {target: source}.
    # Safetensors deduplication may keep either side; if the partner is in the
    # checkpoint the missing one will be re-tied after load, so it's not a
    # real missing key.
    tied_keys = getattr(model, "all_tied_weights_keys", None) or {}
    for target, source in tied_keys.items():
        partner_present = (target in checkpoint_keys) or (source in checkpoint_keys)
        if partner_present:
            missing.discard(target)
            missing.discard(source)

    # Remove keys matching model-declared ignore patterns.
    raw_patterns = list(getattr(model, "_keys_to_ignore_on_load_missing", None) or [])
    if raw_patterns:
        ignore_re = re.compile("|".join(rf"({p})" for p in raw_patterns))
        missing = {k for k in missing if not ignore_re.search(k)}

    return sorted(missing)


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
    process_group: Optional[dist.ProcessGroup] = None,
) -> list[str]:
    """Load HF safetensors weights into an already-materialized model via DCP.

    The model must have been materialized (``to_empty``) before calling this.

    Returns a list of *required* missing key names -- keys present in
    ``model.state_dict()`` but absent from the checkpoint, after filtering out
    tied-weight aliases and known-ignorable patterns (e.g. rotary inv_freq).
    """
    # Resolve HF repo id or local path into a local directory.
    hf_dir = os.path.expanduser(model_name_or_path)
    if not os.path.isdir(hf_dir):
        from huggingface_hub import snapshot_download

        hf_dir = snapshot_download(repo_id=model_name_or_path, repo_type="model")


    if not any(f.endswith(".safetensors") for f in os.listdir(hf_dir)):
        raise FileNotFoundError(f"No .safetensors files found under: {hf_dir}")

    checkpoint_keys = _read_safetensors_keys(hf_dir)
    state_dict = model.state_dict()

    dcp.load(
        state_dict,
        storage_reader=HuggingFaceStorageReader(path=hf_dir),
        planner=DefaultLoadPlanner(allow_partial_load=True),
        process_group=process_group,
    )

    # NOTE: _fixup_after_load (tied embeddings, rotary inv_freq, reward norm
    # buffers) is intentionally NOT called here.  The strategy layer calls
    # _post_load_fixup() after this function returns so that the same fixup
    # path is shared with the DCP-resume flow.

    if dist.is_initialized() and process_group is not None:
        dist.barrier(group=process_group)

    return _get_required_missing_hf_keys(model, checkpoint_keys, set(state_dict.keys()))


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
    model_state = get_model_state_dict(model, options=StateDictOptions(full_state_dict=False, cpu_offload=True))

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

        # Ensure value-head metadata is in config before saving.
        # Runtime normalization stats (mean/std/normalize_reward) are already
        # on config -- they are written there directly by rm_trainer.
        config = getattr(model, "config", None)
        if config is not None:
            config.value_head_prefix = "score"
            if hasattr(config, "auto_map") and isinstance(config.auto_map, dict):
                config.auto_map = {k: v for k, v in config.auto_map.items() if k is not None}
            if hasattr(config, "save_pretrained"):
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
    os.makedirs(save_dir, exist_ok=True)
    if dist.is_initialized():
        dist.barrier(group=process_group)

    unwrapped = unwrap_fn(model)

    # Stateful wrapper for DCP — only model + optimizer tensors.
    class AppState(Stateful):
        def state_dict(self):
            model_state, optim_state = get_state_dict(unwrapped, [optimizer] if optimizer else [])
            return {"model": model_state, "optimizers": optim_state}

        def load_state_dict(self, _):
            raise RuntimeError("Use load_dcp_checkpoint instead")

    if dist.is_initialized():
        dist.barrier(group=process_group)

    with warnings.catch_warnings():
        # `warnings.filterwarnings` expects `category` to be a Warning subclass,
        # not a tuple (Python 3.12 asserts this).
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        dcp.save({"app": AppState()}, checkpoint_id=save_dir, process_group=process_group)

    if dist.is_initialized():
        dist.barrier(group=process_group)

    if is_rank_0:
        # Save non-tensor extras on rank-0 as a small sidecar file.
        # DCP only handles tensor leaves reliably; Python objects (dicts,
        # scalars) are better served by a simple torch.save.
        config = getattr(unwrapped, "config", None)
        if config is not None and hasattr(config, "normalize_reward"):
            runtime_state = {
                "normalize_reward": config.normalize_reward,
                "mean": float(getattr(config, "mean", 0.0)),
                "std": float(getattr(config, "std", 1.0)),
            }
        else:
            runtime_state = {}

        torch.save(
            {
                "client_state": dict(client_state or {}),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "runtime_state": runtime_state,
            },
            os.path.join(save_dir, "extra_state.pt"),
        )
        # Mark checkpoint as complete.
        with open(os.path.join(save_dir, "STABLE"), "w") as f:
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

    unwrapped = unwrap_fn(model)
    should_load_optimizer = load_optimizer_states and optimizer and not load_module_only
    should_load_scheduler = load_lr_scheduler_states and scheduler and not load_module_only

    # Stateful wrapper for loading.
    class AppState(Stateful):
        def __init__(self):
            model_state, optim_state = get_state_dict(unwrapped, [optimizer] if should_load_optimizer else [])
            self._template = {"model": model_state, "optimizers": optim_state}

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
        dcp.load({"app": app}, checkpoint_id=load_dir, process_group=process_group)

    # Load non-tensor extras from rank-0 sidecar file and broadcast.
    extra_path = os.path.join(load_dir, "extra_state.pt")
    is_rank0 = not dist.is_initialized() or dist.get_rank() == 0
    extras = torch.load(extra_path, map_location="cpu") if is_rank0 and os.path.isfile(extra_path) else None
    if dist.is_initialized():
        obj_list = [extras]
        dist.broadcast_object_list(obj_list, src=0, group=process_group)
        extras = obj_list[0]

    if not isinstance(extras, dict):
        if not load_module_only:
            raise FileNotFoundError(f"Full resume requires a valid extra_state.pt at: {extra_path}")
        extras = {}

    # Restore reward normalization stats to config and buffers.
    runtime_state = extras.get("runtime_state") or {}
    if runtime_state and hasattr(unwrapped, "config"):
        config = unwrapped.config
        for key in ("normalize_reward", "mean", "std"):
            if key in runtime_state:
                setattr(config, key, runtime_state[key])
        unwrapped.normalize_reward = config.normalize_reward
    if hasattr(unwrapped, "reset_buffers"):
        unwrapped.reset_buffers()

    if should_load_scheduler and scheduler is not None:
        sched_state = extras.get("scheduler")
        if sched_state is None:
            raise RuntimeError(f"Full resume expects scheduler state in extra_state.pt: {extra_path}")
        scheduler.load_state_dict(sched_state)

    return extras.get("client_state", {})
