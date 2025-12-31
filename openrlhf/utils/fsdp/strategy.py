"""
FSDP2 distributed training strategy for OpenRLHF.

Device mesh layout: (dp, cp, tp)
- dp: data parallelism (FSDP sharding)
- cp: context parallelism (ring attention)
- tp: tensor parallelism
"""

import os
from abc import ABC
from collections import defaultdict
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffloadPolicy, FSDPModule, MixedPrecisionPolicy, fully_shard
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import enable_full_determinism, set_seed

from openrlhf.models.ring_attn_utils import get_ring_attn_group, set_ring_attn_group
from openrlhf.utils import convert_to_dtype
from openrlhf.utils.distributed_sampler import DistributedSampler

from . import MESH_DIM_CP, MESH_DIM_DP, MESH_DIM_TP
from .checkpoint import load_distributed_checkpoint, load_hf_model, save_distributed_checkpoint, save_hf_model
from .utils import (
    barrier,
    clip_grad_norm_dtensor,
    get_runtime_metadata,
    move_optimizer_state,
    moving_average_fsdp,
    unwrap_actor,
)


class FSDP2Strategy(ABC):
    """FSDP2 strategy with DP/CP/TP support."""

    def __init__(
        self, seed=42, full_determinism=False, max_norm=0.0, micro_train_batch_size=1, train_batch_size=1, args=None
    ):
        self.args = args
        self.seed, self.full_determinism, self.max_norm = seed, full_determinism, max_norm
        self.train_batch_size = train_batch_size
        self.micro_train_batch_size = micro_train_batch_size

        # Parallelism config
        self.ring_attn_size = max(1, int(getattr(args, "ring_attn_size", 1) or 1))
        self.tp_size = max(1, int(getattr(args, "ds_tensor_parallel_size", 1) or 1))
        self.ds_tensor_parallel_size = self.tp_size  # backward compat

        # FSDP config
        self.precision = getattr(args, "precision", "bf16")
        self.fsdp2_offload = getattr(args, "fsdp2_offload", "none")
        self.fsdp2_offload_pin_memory = getattr(args, "fsdp2_cpu_offload_pin_memory", True)
        self.fsdp2_reshard_after_forward = getattr(args, "fsdp2_reshard_after_forward", True)
        self.sequence_parallel = getattr(args, "sequence_parallel", False)

        # State
        self.is_rlhf = False
        self.time_steps = defaultdict(int)
        self.mesh = None

    # -------------------------------------------------------------------------
    # Distributed Setup
    # -------------------------------------------------------------------------

    def setup_distributed(self, timeout=timedelta(minutes=60)):
        """Initialize distributed environment."""
        enable_full_determinism(self.seed) if self.full_determinism else set_seed(self.seed)

        if self.args.local_rank != -1:
            os.environ["LOCAL_RANK"] = str(self.args.local_rank)
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if local_rank >= 0:
            torch.cuda.set_device(local_rank)

        dist.init_process_group(timeout=timeout)
        self.world_size = dist.get_world_size()

        # Validate and compute parallelism sizes
        parallel_factor = self.ring_attn_size * self.tp_size
        assert (
            self.world_size % parallel_factor == 0
        ), f"world_size({self.world_size}) not divisible by ring_attn*tp({parallel_factor})"
        self.dp_size = self.world_size // parallel_factor

        # Create mesh: (dp, cp, tp) - only include dimensions > 1
        dims = [(MESH_DIM_DP, self.dp_size)]
        if self.ring_attn_size > 1:
            dims.append((MESH_DIM_CP, self.ring_attn_size))
        if self.tp_size > 1:
            dims.append((MESH_DIM_TP, self.tp_size))
        names, shape = zip(*dims)
        self.mesh = init_device_mesh("cuda", shape, mesh_dim_names=names)

        # Setup ring attention
        self.ring_attn_rank = 0
        set_ring_attn_group(None)
        if self.ring_attn_size > 1:
            cp_group = self.mesh[MESH_DIM_CP].get_group()
            self.ring_attn_rank = dist.get_rank(group=cp_group)
            set_ring_attn_group(cp_group)
            from ring_flash_attn import substitute_hf_flash_attn

            substitute_hf_flash_attn(cp_group, getattr(self.args, "ring_head_stride", 1))

        # Gradient accumulation
        effective_batch = self.micro_train_batch_size * self.world_size
        self.accumulated_gradient = (
            1 if getattr(self.args, "use_dynamic_batch", False) else max(1, self.train_batch_size // effective_batch)
        )

        self._log(
            f"world={self.world_size} dp={self.dp_size} cp={self.ring_attn_size} "
            f"tp={self.tp_size} grad_accum={self.accumulated_gradient}"
        )

    @property
    def ring_attn_group(self):
        return get_ring_attn_group()

    # -------------------------------------------------------------------------
    # Model Preparation
    # -------------------------------------------------------------------------

    def prepare(self, *models, is_rlhf=False):
        """Apply TP + FSDP to models."""
        self.is_rlhf = is_rlhf
        results = [self._wrap(m) if m else None for m in models]
        return results[0] if len(results) == 1 else results

    def _wrap(self, model):
        """Wrap model with TP + FSDP, preserving Actor wrapper."""
        inner, is_actor = unwrap_actor(model), model
        is_actor = inner is not model

        # TP before FSDP
        if self.tp_size > 1 and MESH_DIM_TP in self.mesh.mesh_dim_names:
            from .tp import apply_tensor_parallel

            self._log(f"Applying TP (size={self.tp_size})")
            inner = apply_tensor_parallel(
                inner, self.mesh[MESH_DIM_TP], sequence_parallel=self.sequence_parallel, validate=True
            )

        # FSDP
        inner = self._apply_fsdp(inner)

        if is_actor:
            model.model = inner
            return model
        return inner

    def _apply_fsdp(self, model):
        """Apply FSDP2 sharding."""
        mesh = self.mesh[MESH_DIM_DP]
        mp = (
            None
            if self.precision == "fp32"
            else MixedPrecisionPolicy(
                param_dtype=convert_to_dtype(self.precision),
                reduce_dtype=torch.float32,
                cast_forward_inputs=True,
            )
        )
        offload = CPUOffloadPolicy(pin_memory=self.fsdp2_offload_pin_memory) if self.fsdp2_offload == "cpu" else None

        # Shard transformer layers
        layer_cls = getattr(model, "_no_split_modules", None) or []
        layers = [
            m
            for m in model.modules()
            if m.__class__.__name__ in layer_cls
            or (
                isinstance(m, nn.Embedding)
                and not getattr(getattr(model, "config", None), "tie_word_embeddings", True)
            )
        ]

        for i, layer in enumerate(layers):
            if not isinstance(layer, FSDPModule):
                fully_shard(
                    layer,
                    mesh=mesh,
                    mp_policy=mp,
                    offload_policy=offload,
                    reshard_after_forward=self.fsdp2_reshard_after_forward and i < len(layers) - 1,
                )

        # Shard root
        if not isinstance(model, FSDPModule):
            fully_shard(model, mesh=mesh, mp_policy=mp, offload_policy=offload, reshard_after_forward=False)
        return model

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    def create_optimizer(self, model, **kwargs):
        """Create AdamW optimizer."""
        if "foreach" not in kwargs and self.tp_size > 1 and self.dp_size <= 1:
            kwargs["foreach"] = False
        return optim.AdamW(unwrap_actor(model).parameters(), **kwargs)

    def backward(self, loss, model, optimizer, **kwargs):
        """Backward with gradient accumulation and deferred sync."""
        if self.accumulated_gradient > 1:
            loss = loss / self.accumulated_gradient

        inner = unwrap_actor(model)
        if isinstance(inner, FSDPModule) and self.accumulated_gradient > 1:
            is_final = (
                self.time_steps.get(f"step_{kwargs.get('name', 'model')}", 0) + 1
            ) % self.accumulated_gradient == 0
            inner.set_requires_gradient_sync(is_final)

        loss.backward()

    def optimizer_step(self, optimizer, model, scheduler, name="model", **kwargs):
        """Optimizer step with gradient accumulation."""
        key = f"step_{name}"
        self.time_steps[key] += 1
        if self.time_steps[key] % self.accumulated_gradient != 0:
            return

        if self.max_norm > 0:
            clip_grad_norm_dtensor(unwrap_actor(model), self.max_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if scheduler:
            scheduler.step()

    def moving_average(self, model, model_ema, beta=0.992, device="cpu"):
        """Update EMA model."""
        self.time_steps["ema"] += 1
        if self.time_steps["ema"] % max(1, self.accumulated_gradient) == 0:
            moving_average_fsdp(model, model_ema, beta, device)

    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------

    def setup_dataloader(
        self,
        dataset,
        batch_size,
        pin_memory=False,
        shuffle=True,
        collate_fn=None,
        drop_last=True,
        sampler=None,
        consumed_samples=0,
    ):
        """Create distributed dataloader."""
        if sampler is None and dist.is_initialized():
            dp_group = self.mesh[MESH_DIM_DP].get_group() if self.mesh else None
            sampler = DistributedSampler(
                dataset,
                num_replicas=dist.get_world_size(dp_group) if dp_group else dist.get_world_size(),
                rank=dist.get_rank(dp_group) if dp_group else dist.get_rank(),
                shuffle=shuffle,
                seed=self.seed,
                drop_last=drop_last,
                consumed_samples=consumed_samples,
            )

        return StatefulDataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=drop_last,
            shuffle=shuffle if sampler is None else False,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
        )

    # -------------------------------------------------------------------------
    # State Management
    # -------------------------------------------------------------------------

    def offload_states(self, model, optimizer=None):
        """Offload to CPU."""
        for m in unwrap_actor(model).modules():
            if isinstance(m, FSDPModule):
                m.reshard()
        if optimizer:
            move_optimizer_state(optimizer, torch.device("cpu"))
        torch.cuda.empty_cache()
        barrier()

    def reload_states(self, model, optimizer=None):
        """Reload to GPU."""
        device = torch.device("cuda", torch.cuda.current_device())
        unwrap_actor(model).to(device)
        if optimizer:
            move_optimizer_state(optimizer, device)
        torch.cuda.synchronize()
        barrier()

    # -------------------------------------------------------------------------
    # Checkpointing
    # -------------------------------------------------------------------------

    def save_model(self, model, tokenizer, output_dir, **kwargs):
        save_hf_model(
            model, tokenizer, output_dir, self.is_rank_0(), unwrap_actor, get_runtime_metadata(self), **kwargs
        )

    def load_model(self, model, path, **kwargs):
        load_hf_model(model, path, unwrap_actor, **kwargs)

    def save_ckpt(self, model, save_dir, tag=None, **kwargs):
        save_distributed_checkpoint(model, save_dir, tag, unwrap_actor, self.is_rank_0(), **kwargs)

    def load_ckpt(self, model, load_dir, **kwargs):
        return load_distributed_checkpoint(model, load_dir, unwrap_actor, **kwargs)

    # -------------------------------------------------------------------------
    # Communication
    # -------------------------------------------------------------------------

    def _dp_group(self):
        return self.mesh[MESH_DIM_DP].get_group() if self.mesh else None

    def all_reduce(self, data, op="mean"):
        """All-reduce across DP group."""
        if isinstance(data, dict):
            return {k: self.all_reduce(v, op) for k, v in data.items()}

        group, size = self._dp_group(), dist.get_world_size(self._dp_group()) if dist.is_initialized() else 1
        is_tensor, on_cpu = isinstance(data, torch.Tensor), False
        t = data if is_tensor else torch.tensor(data)
        if t.device.type == "cpu":
            on_cpu, t = True, t.cuda()

        dist.all_reduce(
            t, op={"mean": dist.ReduceOp.SUM, "max": dist.ReduceOp.MAX, "sum": dist.ReduceOp.SUM}[op], group=group
        )
        if op == "mean":
            t = t / size
        return (t.cpu() if on_cpu else t) if is_tensor else (t.cpu() if on_cpu else t).item()

    def all_gather(self, data):
        """All-gather across DP group."""
        group, size = self._dp_group(), dist.get_world_size(self._dp_group()) if dist.is_initialized() else 1
        is_tensor = isinstance(data, torch.Tensor)
        t = data if is_tensor else torch.tensor(data)
        on_cpu = t.device.type == "cpu"

        out = [torch.zeros_like(t).cuda() for _ in range(size)]
        dist.all_gather(out, t.cuda(), group=group)
        result = torch.cat(out)
        return (result.cpu() if on_cpu else result) if is_tensor else result.tolist()

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _log(self, msg):
        if self.is_rank_0():
            print(f"[FSDP2] {msg}")

    def print(self, *msg):
        if self.is_rank_0():
            print(*msg)

    def is_rank_0(self):
        return not dist.is_initialized() or dist.get_rank() == 0

    def get_rank(self):
        return dist.get_rank() if dist.is_initialized() else 0

    def get_world_size(self):
        return dist.get_world_size() if dist.is_initialized() else 1

    # DeepSpeed compatibility stubs
    def get_ds_train_config(self, is_actor=False):
        return None

    def get_ds_eval_config(self, offload=False):
        return None
