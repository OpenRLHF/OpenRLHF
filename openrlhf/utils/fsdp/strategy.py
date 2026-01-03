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


from .checkpoint import load_distributed_checkpoint, load_hf_model, save_distributed_checkpoint, save_hf_model
from .utils import (
    get_runtime_metadata,
    load_fsdp_model_to_gpu,
    move_optimizer_state,
    moving_average_fsdp,
    offload_fsdp_model_to_cpu,
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
        self.fsdp2_cpu_offload = getattr(args, "fsdp2_cpu_offload", False)
        self.fsdp2_reshard_after_forward = getattr(args, "fsdp2_reshard_after_forward", True)
        self.sequence_parallel = getattr(args, "sequence_parallel", False)

        # CPUOffloadPolicy and manual offload are mutually exclusive (ref: slime)
        # When fsdp2_cpu_offload is enabled, FSDP2 manages CPU offload automatically,
        # so deepspeed_enable_sleep (manual offload) should be disabled.
        if self.fsdp2_cpu_offload and getattr(args, "deepspeed_enable_sleep", False):
            args.deepspeed_enable_sleep = False
            print("[FSDP2] Warning: deepspeed_enable_sleep disabled because fsdp2_cpu_offload is enabled")

        # State
        self.time_steps = defaultdict(int)
        self.mesh = None
        self._gloo_group = None  # Gloo group for CPU-safe barriers

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

        # Initialize Gloo group for CPU-safe barriers (ref: slime)
        # NCCL barriers require GPU tensors, which fail when model is offloaded to CPU.
        # Gloo works with CPU tensors and is safe to use after model offload.
        self._gloo_group = dist.new_group(backend="gloo")

        # Validate and compute parallelism sizes
        parallel_factor = self.ring_attn_size * self.tp_size
        assert (
            self.world_size % parallel_factor == 0
        ), f"world_size({self.world_size}) not divisible by ring_attn*tp({parallel_factor})"
        self.dp_size = self.world_size // parallel_factor

        # Create fixed 3D mesh: (dp, cp, tp) - always include all dimensions even if size=1
        # This ensures all parameters use the same mesh structure, avoiding mesh mismatch issues
        # in operations like clip_grad_norm that aggregate across all parameters
        self.mesh = init_device_mesh(
            "cuda",
            (self.dp_size, self.ring_attn_size, self.tp_size),
            mesh_dim_names=("dp", "cp", "tp"),
        )

        # Pre-create process groups for gradient norm computation (aligned with Automodel)
        # Use _flatten to create 1D mesh from 2D dp+cp submesh
        self.mesh["dp", "cp"]._flatten(mesh_dim_name="dp_cp")
        self.dp_cp_group = self.mesh["dp_cp"].get_group()
        self.tp_group = self.mesh["tp"].get_group() if self.tp_size > 1 else None

        # Setup ring attention
        set_ring_attn_group(None)
        if self.ring_attn_size > 1:
            cp_group = self.mesh["cp"].get_group()
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

    def prepare(self, *models):
        """Apply TP + FSDP to models."""
        results = [self._wrap(m) if m else None for m in models]
        return results[0] if len(results) == 1 else results

    def prepare_ref(self, model):
        """Apply FSDP with forced CPUOffloadPolicy for Reference model.

        Reference models are only used for inference during training, so we always
        enable CPUOffloadPolicy to save GPU memory. This is more efficient than
        manually calling model.cpu()/model.cuda().
        """
        return self._wrap(model, force_cpu_offload=True)

    def _wrap(self, model, force_cpu_offload=False):
        """Wrap model with TP + FSDP, preserving Actor wrapper.

        Args:
            model: Model to wrap
            force_cpu_offload: If True, force CPUOffloadPolicy (for Reference models)
        """
        inner, is_actor = self._unwrap_model(model), model
        is_actor = inner is not model

        # TP before FSDP
        print(f"[FSDP2 wrap_model] tp_size={self.tp_size}, mesh_dim_names={self.mesh.mesh_dim_names}")
        if self.tp_size > 1:
            from .tp import apply_tensor_parallel

            self._log(f"Applying TP (size={self.tp_size})")
            print(f"[FSDP2 wrap_model] TP mesh: {self.mesh['tp']}")
            inner = apply_tensor_parallel(
                inner,
                self.mesh["tp"],
                sequence_parallel=self.sequence_parallel,
                validate=True,
                ring_attn_group=self.ring_attn_group if self.ring_attn_size > 1 else None,
            )
        else:
            print(f"[FSDP2 wrap_model] Skipping TP: tp_size={self.tp_size}")

        # FSDP (force_cpu_offload overrides self.fsdp2_cpu_offload)
        inner = self._apply_fsdp(inner, force_cpu_offload=force_cpu_offload)

        if is_actor:
            model.model = inner
            return model
        return inner

    def _apply_fsdp(self, model, force_cpu_offload=False):
        """Apply FSDP2 sharding.

        Args:
            model: Model to shard
            force_cpu_offload: If True, force CPUOffloadPolicy regardless of self.fsdp2_cpu_offload
        """
        mesh = self.mesh["dp"]
        mp = (
            None
            if self.precision == "fp32"
            else MixedPrecisionPolicy(
                param_dtype=convert_to_dtype(self.precision),
                reduce_dtype=torch.float32,
                cast_forward_inputs=True,
            )
        )
        use_cpu_offload = force_cpu_offload or self.fsdp2_cpu_offload
        offload = CPUOffloadPolicy(pin_memory=True) if use_cpu_offload else None

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
        return optim.AdamW(self._unwrap_model(model).parameters(), **kwargs)

    def backward(self, loss, model, optimizer, name="model", **kwargs):
        """Backward with gradient accumulation and deferred sync."""
        if self.accumulated_gradient > 1:
            loss = loss / self.accumulated_gradient

        inner = self._unwrap_model(model)
        if isinstance(inner, FSDPModule) and self.accumulated_gradient > 1:
            key = f"step_{name}"
            is_final = (self.time_steps.get(key, 0) + 1) % self.accumulated_gradient == 0
            inner.set_requires_gradient_sync(is_final)

        loss.backward()

    def _get_grad_norm(self, model, norm_type=2.0, dtype=torch.float32):
        """Calculate gradient norm across all parallel groups.

        Reference: Automodel/nemo_automodel/components/distributed/grad_utils.py
        """
        inner = self._unwrap_model(model)
        parameters = [p for p in inner.parameters() if p.grad is not None]

        if len(parameters) == 0:
            return 0.0

        # Get local gradients (convert DTensor to local tensor)
        grads = [
            (p.grad.detach().to_local() if isinstance(p.grad, DTensor) else p.grad.detach()).to(dtype)
            for p in parameters
        ]

        norm_type = float(norm_type)

        if norm_type == float("inf"):
            total_norm = max(g.abs().max().item() for g in grads)
            total_norm = torch.tensor([total_norm], dtype=torch.float, device="cuda")
            dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=self.dp_cp_group)
            if self.tp_group:
                dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=self.tp_group)
            return total_norm[0].item()

        # L2 or other norms
        total_norm = sum(torch.norm(g, norm_type) ** norm_type for g in grads)
        total_norm = torch.tensor(total_norm, device="cuda")
        dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=self.dp_cp_group)
        if self.tp_group:
            dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=self.tp_group)
        return total_norm.item() ** (1.0 / norm_type)

    def _clip_grad_norm(self, model, max_norm, norm_type=2.0):
        """Clip gradients by total norm (aligned with Automodel)."""
        total_norm = self._get_grad_norm(model, norm_type)
        if total_norm == 0.0:
            return total_norm

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1.0:
            for p in self._unwrap_model(model).parameters():
                if p.grad is not None:
                    g = p.grad.detach()
                    (g.to_local() if isinstance(g, DTensor) else g).mul_(clip_coef)

        return total_norm

    def optimizer_step(self, optimizer, model, scheduler, name="model", **kwargs):
        """Optimizer step with gradient accumulation."""
        key = f"step_{name}"
        self.time_steps[key] += 1
        if self.time_steps[key] % self.accumulated_gradient != 0:
            return

        if self.max_norm > 0:
            # Use DTensor-compatible gradient clipping for TP+FSDP
            # Standard clip_grad_norm_ fails when params have different meshes
            self._clip_grad_norm(model, self.max_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if scheduler:
            scheduler.step()

    def moving_average(self, model, model_ema, beta=0.992, device="cpu"):
        """Update EMA model."""
        self.time_steps["ema"] += 1
        if self.time_steps["ema"] % max(1, self.accumulated_gradient) == 0:
            moving_average_fsdp(model, model_ema, self._unwrap_model, beta, device)

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
            dp_group = self.mesh["dp"].get_group() if self.mesh else None
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
        """Offload optimizer states to CPU, optionally offload model too.

        For FSDP2 with vLLM sleep mode:
        - Optimizer states are offloaded to CPU
        - Model params stay on GPU (sharded) for weight sync with vLLM
        - After weight sync, call offload_model() to offload model to CPU

        Note: When fsdp2_cpu_offload is enabled, optimizer runs on CPU and states
        are already in CPU memory, so optimizer offload is skipped.
        """
        # Skip if using CPUOffloadPolicy - optimizer state is already on CPU
        if self.fsdp2_cpu_offload:
            if self.is_rank_0():
                print("[FSDP2 offload_states] Skipping - CPUOffloadPolicy already manages state on CPU")
            return

        if optimizer:
            move_optimizer_state(optimizer, torch.device("cpu"))

        torch.cuda.empty_cache()
        # Use Gloo barrier instead of NCCL - NCCL may fail when model is on CPU (ref: slime)
        if dist.is_initialized() and self._gloo_group is not None:
            dist.barrier(group=self._gloo_group)
        torch.cuda.synchronize()

    def offload_model(self, model):
        """Offload model to CPU (separate from optimizer offload).

        This should be called AFTER weight sync with vLLM to free GPU memory
        for vLLM's wake_up operation.
        """
        inner = self._unwrap_model(model)
        already_on_cpu = next(inner.parameters()).device.type == "cpu"
        if not already_on_cpu:
            offload_fsdp_model_to_cpu(inner, empty_cache=True)
            if self.is_rank_0():
                print("[FSDP2 offload_model] Model offloaded to CPU")
        else:
            if self.is_rank_0():
                print("[FSDP2 offload_model] Model already on CPU (skip)")

        # Use Gloo barrier - model is now on CPU, NCCL requires GPU tensors (ref: slime)
        if self._gloo_group is not None:
            dist.barrier(group=self._gloo_group)
        torch.cuda.synchronize()

    def reload_model(self, model):
        """Reload model to GPU (separate from optimizer reload).

        This should be called BEFORE training when model was offloaded.
        """
        inner = self._unwrap_model(model)
        already_on_cuda = next(inner.parameters()).device.type == "cuda"
        if not already_on_cuda:
            load_fsdp_model_to_gpu(inner, device_id=torch.cuda.current_device())
            if self.is_rank_0():
                print("[FSDP2 reload_model] Model reloaded to GPU")
        else:
            if self.is_rank_0():
                print("[FSDP2 reload_model] Model already on GPU (skip)")

        torch.cuda.synchronize()
        # Use Gloo barrier for consistency (ref: slime)
        if self._gloo_group is not None:
            dist.barrier(group=self._gloo_group)

    def reload_states(self, model, optimizer=None):
        """Reload model params and optimizer states to GPU.

        This must be called before training to restore the states from CPU.
        Also reloads the model if it was offloaded via offload_model().

        Note: When fsdp2_cpu_offload is enabled, optimizer runs on CPU and states
        are already in CPU memory, so optimizer reload is skipped.
        """
        # Skip if using CPUOffloadPolicy - optimizer state stays on CPU
        if self.fsdp2_cpu_offload:
            if self.is_rank_0():
                print("[FSDP2 reload_states] Skipping - CPUOffloadPolicy manages state on CPU")
            return

        device = torch.device("cuda", torch.cuda.current_device())

        # Reload model to GPU if it was offloaded
        inner = self._unwrap_model(model)
        if next(inner.parameters()).device.type == "cpu":
            load_fsdp_model_to_gpu(inner, device_id=torch.cuda.current_device())
            if self.is_rank_0():
                print("[FSDP2 reload_states] Model reloaded to GPU")

        # Reload optimizer states
        if optimizer:
            move_optimizer_state(optimizer, device)

        torch.cuda.synchronize()
        # Use Gloo barrier for consistency (ref: slime)
        if self._gloo_group is not None:
            dist.barrier(group=self._gloo_group)

    # -------------------------------------------------------------------------
    # Checkpointing
    # -------------------------------------------------------------------------

    def save_model(self, model, tokenizer, output_dir, **kwargs):
        save_hf_model(
            model, tokenizer, output_dir, self.is_rank_0(), self._unwrap_model, get_runtime_metadata(self), **kwargs
        )

    def load_model(self, model, path, **kwargs):
        load_hf_model(model, path, self._unwrap_model, **kwargs)

    def save_ckpt(self, model, save_dir, tag=None, **kwargs):
        # Use a Gloo process group for checkpoint collectives. This is more
        # robust when parameters or optimizer states may live on CPU (e.g.,
        # CPUOffloadPolicy / manual offload), since NCCL requires CUDA tensors.
        save_distributed_checkpoint(
            model,
            save_dir,
            tag,
            self._unwrap_model,
            self.is_rank_0(),
            process_group=self._gloo_group,
            **kwargs,
        )

    def load_ckpt(self, model, load_dir, **kwargs):
        return load_distributed_checkpoint(
            model, load_dir, self._unwrap_model, process_group=self._gloo_group, **kwargs
        )

    # -------------------------------------------------------------------------
    # Communication
    # -------------------------------------------------------------------------

    def _dp_group(self):
        return self.mesh["dp"].get_group() if self.mesh else None

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
    def _unwrap_model(self, model):
        """Unwrap model (compatible with DeepspeedStrategy)."""
        try:
            from openrlhf.models import Actor

            if isinstance(model, Actor):
                return self._unwrap_model(model.model)
        except ImportError:
            pass
        return model

    def get_ds_train_config(self, is_actor=False):
        return None

    def get_ds_eval_config(self, offload=False):
        return None
