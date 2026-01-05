"""
FSDP2 distributed training strategy for OpenRLHF.

Device mesh layout: (dp, cp, tp)
- dp: data parallelism (data sharding across replicas)
- cp: context parallelism (ring attention for long sequences)
- tp: tensor parallelism (parameter sharding within a model)

FSDP sharding uses merged dp_cp mesh:
- FSDP reduce-scatter covers all DP+CP ranks
- This ensures gradients are correctly aggregated across both DP and CP
- Ring Attention only aggregates dK/dV (activation gradients), NOT parameter gradients
- Without merging DP+CP, different CP ranks would have divergent parameters!

Reference: Automodel, torchtitan both merge DP+CP for FSDP mesh.
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
from torch.distributed.tensor import DTensor
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

        # Create 3D mesh: (dp, cp, tp) - always include all dimensions even if size=1
        # This ensures all parameters use the same mesh structure, avoiding mesh mismatch issues
        # in operations like clip_grad_norm that aggregate across all parameters
        self.mesh = init_device_mesh(
            "cuda",
            (self.dp_size, self.ring_attn_size, self.tp_size),
            mesh_dim_names=("dp", "cp", "tp"),
        )

        # Create FSDP mesh by merging DP and CP dimensions
        # This is CRITICAL for Ring Attention correctness:
        # - Ring Attention backward only aggregates dK, dV (activation gradients)
        # - Parameter gradients (dW_q, dW_k, dW_v, etc.) are NOT aggregated by Ring Attention
        # - FSDP's reduce-scatter must cover all DP+CP ranks to aggregate parameter gradients
        # - Without this, different CP ranks would have divergent parameters!
        # Reference: Automodel and torchtitan both use this approach
        self.fsdp_mesh = self.mesh["dp", "cp"]._flatten(mesh_dim_name="dp_cp")

        # Process groups for different purposes:
        # - dp_cp_group: for metrics reduction across all FSDP ranks (DP + CP)
        # - dp_group: for data sampling (only DP, CP ranks share same data)
        # - cp_group: for Ring Attention KV communication
        # Note: tp_group is no longer needed for grad norm (DTensor handles it)
        self.dp_cp_group = self.fsdp_mesh.get_group()
        self.dp_group = self.mesh["dp"].get_group()
        self.cp_group = self.mesh["cp"].get_group()

        # Setup ring attention using CP group
        set_ring_attn_group(None)
        if self.ring_attn_size > 1:
            set_ring_attn_group(self.cp_group)
            from ring_flash_attn import substitute_hf_flash_attn

            substitute_hf_flash_attn(self.cp_group, getattr(self.args, "ring_head_stride", 1))

        # Gradient accumulation
        # Only DP contributes to batch size - CP processes same sequence chunks, TP processes same batch
        # This matches DeepSpeed formula: train_batch_size / (micro_train_batch_size * dp_size)
        effective_batch = self.micro_train_batch_size * self.dp_size
        self.accumulated_gradient = (
            1 if getattr(self.args, "use_dynamic_batch", False) else max(1, self.train_batch_size // effective_batch)
        )

        self._log(
            f"world={self.world_size} dp={self.dp_size} cp={self.ring_attn_size} "
            f"tp={self.tp_size} grad_accum={self.accumulated_gradient} "
            f"fsdp_mesh_size={self.dp_size * self.ring_attn_size}"
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
        print(f"[FSDP2 wrap_model] tp_size={self.tp_size}, mesh_dim_names={self.mesh.mesh_dim_names}, "
              f"fsdp_mesh_size={self.dp_size * self.ring_attn_size} (dp={self.dp_size} × cp={self.ring_attn_size})")
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
        """Apply FSDP2 sharding with merged DP+CP mesh.

        Args:
            model: Model to shard
            force_cpu_offload: If True, force CPUOffloadPolicy regardless of self.fsdp2_cpu_offload

        Note:
            Uses fsdp_mesh (dp_cp flattened) instead of mesh["dp"] to ensure
            FSDP reduce-scatter covers all DP+CP ranks. This is essential for
            Ring Attention because it only aggregates dK/dV, not parameter gradients.
        """
        # Use merged DP+CP mesh for FSDP - critical for Ring Attention correctness
        mesh = self.fsdp_mesh
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
        """Backward with gradient accumulation and deferred sync.

        Note on loss scaling with CP (Context Parallelism):
        - Ring Attention slices the sequence across CP ranks for the forward pass,
          then uses `flash_attn.utils.distributed.all_gather` to rebuild full-sequence tensors
          (log_probs / values / etc.) on EVERY CP rank before loss computation.
        - `all_gather` autograd uses reduce_scatter(SUM) in backward, which introduces a `cp_size`
          factor into the local gradients.
        - FSDP2 then averages gradients across the flattened `dp_cp` mesh, dividing by
          (dp_size * cp_size); the `cp_size` factor from `all_gather` cancels this, yielding an
          effective "dp average, cp sum" without explicitly scaling the loss here.

        MoE aux_loss handling:
        - Trainers combine main_loss and aux_loss before calling backward:
          loss = main_loss + aux_loss * aux_loss_coef
        - aux_loss is computed locally on each CP rank based on the tokens it processes.
        - FSDP2's AVG across (dp_size * cp_size) correctly averages aux_loss across all ranks,
          giving us the global load balance metric we want.
        - No special handling needed here - both losses flow through the same backward().

        Args:
            loss: Combined loss (main_loss + aux_loss * coef if MoE)
            model: Model being trained
            optimizer: Optimizer
            name: Name for gradient sync tracking
        """
        if self.accumulated_gradient > 1:
            loss = loss / self.accumulated_gradient

        inner = self._unwrap_model(model)
        if isinstance(inner, FSDPModule) and self.accumulated_gradient > 1:
            key = f"step_{name}"
            is_final = (self.time_steps.get(key, 0) + 1) % self.accumulated_gradient == 0
            inner.set_requires_gradient_sync(is_final)

        loss.backward()

    @torch.no_grad()
    def _clip_grad_norm(self, model, max_norm, norm_type=2.0):
        """Clip gradients for FSDP2 + TP models.

        With FSDP mesh = dp_cp, parameters are DTensors distributed across multiple
        dimensions. get_total_norm + full_tensor handles aggregation automatically:
        - Shard placements: full_tensor() aggregates across the sharded dimension
        - Replicate placements: full_tensor() returns as-is (already complete)

        This is much simpler than manual all_reduce because DTensor tracks the
        correct aggregation logic based on placements.

        Reference: Automodel's _clip_grad_norm_impl uses the same approach.
        """
        import math

        parameters = [p for p in self._unwrap_model(model).parameters() if p.grad is not None]
        if not parameters:
            return 0.0

        norm_type = float(norm_type)

        # Group parameters by their sharding pattern (mesh + placements)
        # Different sharding patterns must be handled separately for clip_grads_with_norm_
        sharding_groups = {}
        for p in parameters:
            if isinstance(p, DTensor):
                key = (id(p.device_mesh), tuple(str(pl) for pl in p.placements))
            else:
                key = ("regular", "regular")
            sharding_groups.setdefault(key, []).append(p)

        # Compute norm for each sharding group
        group_norms = []
        for group_params in sharding_groups.values():
            grads = [p.grad for p in group_params]
            norm = torch.nn.utils.get_total_norm(grads, norm_type, error_if_nonfinite=False)

            # Convert DTensor norm to regular tensor via full_tensor()
            # This automatically handles aggregation across all mesh dimensions
            if isinstance(norm, DTensor):
                norm = norm.full_tensor()

            norm = norm.float().cuda().clone().detach()
            group_norms.append(norm)

        # Combine norms across groups
        if len(group_norms) == 0:
            total_norm = torch.tensor(0.0, device="cuda")
        elif len(group_norms) == 1:
            total_norm = group_norms[0]
        else:
            if math.isinf(norm_type):
                total_norm = torch.stack(group_norms).max()
            else:
                # Combine p-norms: (sum of p-th powers)^(1/p)
                total_norm = sum(gn ** norm_type for gn in group_norms) ** (1.0 / norm_type)

        # Clip gradients for each sharding group separately
        for group_params in sharding_groups.values():
            torch.nn.utils.clip_grads_with_norm_(group_params, max_norm, total_norm)

        return total_norm.item()

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
        """Create distributed dataloader.

        Note:
            Data is sharded only across DP dimension, not CP.
            CP ranks within the same DP group see the same data but process different
            sequence chunks. This is essential for Ring Attention to work correctly.
        """
        if sampler is None and dist.is_initialized():
            # Use dp_group for data sharding - CP ranks share the same data
            # This ensures all CP ranks in the same DP group see identical samples
            dp_group = self.dp_group if hasattr(self, 'dp_group') and self.dp_group else None
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
        """Return DP process group (for data-related operations, not FSDP)."""
        return self.dp_group if hasattr(self, 'dp_group') and self.dp_group else None

    def all_reduce(self, data, op="mean", with_context_parallel=True):
        """All-reduce across DP group (optionally including CP).

        Args:
            data: Data to reduce (tensor, scalar, or dict)
            op: Reduction operation ("mean", "max", "sum")
            with_context_parallel: If True, reduce across both DP and CP (for metrics).
                                   If False, reduce across DP only (for data partitioning).
                                   Follows slime's convention.
        """
        if isinstance(data, dict):
            return {k: self.all_reduce(v, op, with_context_parallel) for k, v in data.items()}

        # Use dp_cp_group for metrics (like slime's with_context_parallel=True)
        # Use dp_group only for data-related operations
        if with_context_parallel and self.ring_attn_size > 1:
            group = self.dp_cp_group
            size = self.dp_size * self.ring_attn_size
        else:
            group = self._dp_group()
            size = dist.get_world_size(self._dp_group()) if dist.is_initialized() else 1
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
