"""
FSDP2 distributed training strategy for OpenRLHF.

Provides FSDP2-based distributed training with support for:
- Data Parallelism (FSDP sharding)
- Context Parallelism (ring attention)
- Tensor Parallelism (native PyTorch TP)

Device mesh layout: (dp, cp, tp)
- dp: outer dimension, lowest communication frequency
- cp: middle dimension, for ring attention
- tp: inner dimension, highest communication frequency
"""

import os
from abc import ABC
from collections import defaultdict
from datetime import timedelta
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    FSDPModule,
    MixedPrecisionPolicy,
    fully_shard,
)
from torch.optim import Optimizer
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import enable_full_determinism, set_seed

from openrlhf.models.ring_attn_utils import get_ring_attn_group, set_ring_attn_group
from openrlhf.utils import convert_to_dtype
from openrlhf.utils.distributed_sampler import DistributedSampler

from . import MESH_DIM_CP, MESH_DIM_DP, MESH_DIM_TP
from .checkpoint import (
    load_distributed_checkpoint,
    load_hf_model,
    save_distributed_checkpoint,
    save_hf_model,
)
from .utils import (
    barrier,
    clip_grad_norm_dtensor,
    get_runtime_metadata,
    move_optimizer_state,
    moving_average_fsdp,
    unwrap_actor,
)


class FSDP2Strategy(ABC):
    """FSDP2 distributed training strategy.

    Supports DP, CP (ring attention), and TP with device mesh layout (dp, cp, tp).
    """

    def __init__(
        self,
        seed: int = 42,
        full_determinism: bool = False,
        max_norm: float = 0.0,
        micro_train_batch_size: int = 1,
        train_batch_size: int = 1,
        args=None,
    ) -> None:
        super().__init__()

        # Core config
        self.args = args
        self.seed = seed
        self.full_determinism = full_determinism
        self.max_norm = max_norm
        self.train_batch_size = train_batch_size
        self.micro_train_batch_size = micro_train_batch_size
        self.precision = getattr(args, "precision", "bf16")

        # Parallelism sizes
        self.ring_attn_size = max(1, int(getattr(args, "ring_attn_size", 1) or 1))
        self.tp_size = max(1, int(getattr(args, "ds_tensor_parallel_size", 1) or 1))

        # For backward compatibility
        self.ds_tensor_parallel_size = self.tp_size

        # FSDP2 config
        self.fsdp2_offload = getattr(args, "fsdp2_offload", "none")
        self.fsdp2_cpu_offload_pin_memory = getattr(args, "fsdp2_cpu_offload_pin_memory", True)
        self.fsdp2_reshard_after_forward = getattr(args, "fsdp2_reshard_after_forward", True)
        self.sequence_parallel = getattr(args, "sequence_parallel", False)

        # State
        self.is_rlhf = False
        self.time_steps = defaultdict(int)
        self.fsdp_device_mesh = None

        # Build policies
        self._offload_policy = self._build_offload_policy()
        self._mp_policy = self._build_mp_policy()

    # =========================================================================
    # Initialization
    # =========================================================================

    def _build_offload_policy(self) -> Optional[CPUOffloadPolicy]:
        """Build CPU offload policy."""
        mode = (self.fsdp2_offload or "none").lower()
        if mode == "none":
            return None
        if mode == "cpu":
            return CPUOffloadPolicy(pin_memory=self.fsdp2_cpu_offload_pin_memory)
        raise ValueError(f"Unknown FSDP2 offload mode: {mode}")

    def _build_mp_policy(self) -> Optional[MixedPrecisionPolicy]:
        """Build mixed precision policy."""
        if self.precision == "fp32":
            return None
        return MixedPrecisionPolicy(
            param_dtype=convert_to_dtype(self.precision),
            reduce_dtype=torch.float32,
            output_dtype=torch.float32,
            cast_forward_inputs=True,
        )

    def setup_distributed(self, timeout=timedelta(minutes=60)) -> None:
        """Initialize distributed environment and device mesh."""
        # Determinism
        if self.full_determinism:
            enable_full_determinism(self.seed)
        else:
            set_seed(self.seed)

        # Device setup
        if self.args.local_rank != -1:
            os.environ["LOCAL_RANK"] = str(self.args.local_rank)
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        if local_rank != -1:
            torch.cuda.set_device(local_rank)

        # Process group
        dist.init_process_group(timeout=timeout)
        self.world_size = dist.get_world_size()
        self.ring_attn_rank = 0
        set_ring_attn_group(None)

        # Validate world size
        parallel_factor = self.ring_attn_size * self.tp_size
        if self.world_size % parallel_factor != 0:
            raise ValueError(
                f"world_size({self.world_size}) must be divisible by "
                f"ring_attn_size({self.ring_attn_size}) * tp_size({self.tp_size})"
            )
        self.dp_size = self.world_size // parallel_factor

        # Create mesh and setup parallelism
        self._create_device_mesh()
        if self.ring_attn_size > 1:
            self._setup_ring_attention()
        self._compute_grad_accumulation()

        self._log(
            f"world={self.world_size} dp={self.dp_size} cp={self.ring_attn_size} "
            f"tp={self.tp_size} grad_accum={self.accumulated_gradient}"
        )

    def _create_device_mesh(self) -> None:
        """Create device mesh with layout (dp, cp, tp).

        FSDP shards only on dp dimension,
        CP groups hold full parameter copies for ring_flash_attn.
        """
        cp = self.ring_attn_size if self.ring_attn_size > 1 else 1
        tp = self.tp_size if self.tp_size > 1 else 1

        if cp == 1 and tp == 1:
            shape, names = (self.dp_size,), (MESH_DIM_DP,)
        elif cp == 1:
            shape, names = (self.dp_size, tp), (MESH_DIM_DP, MESH_DIM_TP)
        elif tp == 1:
            shape, names = (self.dp_size, cp), (MESH_DIM_DP, MESH_DIM_CP)
        else:
            shape, names = (self.dp_size, cp, tp), (MESH_DIM_DP, MESH_DIM_CP, MESH_DIM_TP)

        self.fsdp_device_mesh = init_device_mesh("cuda", shape, mesh_dim_names=names)

    def _setup_ring_attention(self) -> None:
        """Setup ring attention using ring_flash_attn."""
        cp_group = self.fsdp_device_mesh[MESH_DIM_CP].get_group()
        self.ring_attn_rank = dist.get_rank(group=cp_group)
        set_ring_attn_group(cp_group)

        try:
            from ring_flash_attn import substitute_hf_flash_attn

            stride = getattr(self.args, "ring_head_stride", 1)
            substitute_hf_flash_attn(cp_group, stride)
        except ImportError as e:
            raise RuntimeError("ring_flash_attn required for CP > 1") from e

    def _compute_grad_accumulation(self) -> None:
        """Compute gradient accumulation steps."""
        effective_batch = self.micro_train_batch_size * self.world_size
        self.accumulated_gradient = max(1, self.train_batch_size // effective_batch)
        if getattr(self.args, "use_dynamic_batch", False):
            self.accumulated_gradient = 1

    # =========================================================================
    # Mesh Access
    # =========================================================================

    def _get_fsdp_mesh(self):
        """Get FSDP mesh (dp only, for ring_flash_attn compatibility)."""
        if self.fsdp_device_mesh is None:
            return None
        return self.fsdp_device_mesh[MESH_DIM_DP]

    def _get_tp_mesh(self):
        """Get TP mesh if enabled."""
        if self.tp_size <= 1 or self.fsdp_device_mesh is None:
            return None
        dims = getattr(self.fsdp_device_mesh, "mesh_dim_names", ())
        return self.fsdp_device_mesh[MESH_DIM_TP] if MESH_DIM_TP in dims else None

    def _get_dp_group(self):
        """Get DP process group for communication."""
        if self.fsdp_device_mesh is None:
            return None
        return self.fsdp_device_mesh[MESH_DIM_DP].get_group()

    @property
    def ring_attn_group(self):
        """Ring attention process group."""
        return get_ring_attn_group()

    # =========================================================================
    # Model Preparation
    # =========================================================================

    def prepare(self, *models, is_rlhf: bool = False):
        """Prepare models for FSDP2 training (TP -> FSDP)."""
        self.is_rlhf = is_rlhf
        results = [self._wrap_model(m) if m else None for m in models]
        return results[0] if len(results) == 1 else results

    def _wrap_model(self, model: nn.Module) -> nn.Module:
        """Apply TP + FSDP to model, preserving Actor wrapper."""
        inner = unwrap_actor(model)
        is_actor = inner is not model

        inner = self._apply_tp(inner)
        inner = self._apply_fsdp(inner)

        if is_actor:
            model.model = inner
            return model
        return inner

    def _apply_tp(self, model: nn.Module) -> nn.Module:
        """Apply tensor parallelism if enabled."""
        tp_mesh = self._get_tp_mesh()
        if tp_mesh is None:
            return model

        from .tp import apply_tensor_parallel

        self._log(f"Applying TP (size={tp_mesh.size()}, SP={self.sequence_parallel})")
        return apply_tensor_parallel(model, tp_mesh, sequence_parallel=self.sequence_parallel, validate=True)

    def _apply_fsdp(self, model: nn.Module) -> nn.Module:
        """Apply FSDP2 wrapping."""
        mesh = self._get_fsdp_mesh()
        if mesh is None:
            raise RuntimeError("Call setup_distributed() first")

        # Shard children first, then root
        self._shard_layers(model, mesh)
        if not isinstance(model, FSDPModule):
            model = fully_shard(
                model,
                mesh=mesh,
                reshard_after_forward=False,
                offload_policy=self._offload_policy,
                mp_policy=self._mp_policy,
            )
        return model

    def _shard_layers(self, model: nn.Module, mesh) -> None:
        """Shard transformer layers with FSDP."""
        layer_classes = getattr(model, "_no_split_modules", None) or []

        to_shard = []
        for name, child in model.named_modules():
            if isinstance(child, FSDPModule):
                continue
            if child.__class__.__name__ in layer_classes:
                to_shard.append(child)
            elif isinstance(child, nn.Embedding):
                cfg = getattr(model, "config", None)
                if cfg and not getattr(cfg, "tie_word_embeddings", True):
                    to_shard.append(child)

        for i, layer in enumerate(to_shard):
            is_last = i == len(to_shard) - 1
            fully_shard(
                layer,
                mesh=mesh,
                reshard_after_forward=self.fsdp2_reshard_after_forward and not is_last,
                offload_policy=self._offload_policy,
                mp_policy=self._mp_policy,
            )

    # =========================================================================
    # Training
    # =========================================================================

    def create_optimizer(self, model, **kwargs) -> Optimizer:
        """Create AdamW optimizer."""
        inner = unwrap_actor(model)
        if "foreach" not in kwargs and self.tp_size > 1 and self.dp_size <= 1:
            kwargs["foreach"] = False
        return optim.AdamW(inner.parameters(), **kwargs)

    def backward(self, loss: torch.Tensor, model: nn.Module, optimizer: Optimizer, **kwargs) -> None:
        """Backward pass with gradient accumulation.

        Uses FSDP2's set_requires_gradient_sync to defer gradient sync
        until the final micro-batch for better performance.
        """
        if self.accumulated_gradient > 1:
            loss = loss / self.accumulated_gradient

        # Defer gradient sync until final micro-batch (optimization from Automodel)
        inner = unwrap_actor(model)
        is_final = self._is_final_micro_batch(kwargs.get("name", "model"))
        if isinstance(inner, FSDPModule) and self.accumulated_gradient > 1:
            inner.set_requires_gradient_sync(is_final)

        loss.backward()

    def _is_final_micro_batch(self, name: str = "model") -> bool:
        """Check if this is the final micro-batch before optimizer step."""
        key = f"micro_step_{name}"
        return (self.time_steps.get(key, 0) + 1) % self.accumulated_gradient == 0

    def optimizer_step(self, optimizer: Optimizer, model: nn.Module, scheduler, name: str = "model", **kwargs) -> None:
        """Optimizer step with gradient accumulation."""
        key = f"micro_step_{name}"
        self.time_steps[key] += 1

        if self.time_steps[key] % self.accumulated_gradient != 0:
            return

        if self.max_norm > 0:
            clip_grad_norm_dtensor(unwrap_actor(model), self.max_norm)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if scheduler:
            scheduler.step()

    # =========================================================================
    # EMA
    # =========================================================================

    def moving_average(self, model, model_ema, beta: float = 0.992, device: str = "cpu") -> None:
        """Update EMA model."""
        self.time_steps["ema"] += 1
        if self.time_steps["ema"] % max(1, self.accumulated_gradient) == 0:
            moving_average_fsdp(model, model_ema, beta, device)

    # =========================================================================
    # Data Loading
    # =========================================================================

    def setup_dataloader(
        self,
        dataset,
        batch_size: int,
        pin_memory: bool = False,
        shuffle: bool = True,
        collate_fn=None,
        drop_last: bool = True,
        sampler=None,
        consumed_samples: int = 0,
    ):
        """Create dataloader with distributed sampler."""
        if sampler is None and dist.is_initialized():
            dp_group = self._get_dp_group()
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

    # =========================================================================
    # State Management
    # =========================================================================

    def offload_states(self, model, optimizer=None) -> None:
        """Offload states to CPU."""
        for m in unwrap_actor(model).modules():
            if isinstance(m, FSDPModule):
                m.reshard()
        if optimizer:
            move_optimizer_state(optimizer, torch.device("cpu"))
        torch.cuda.empty_cache()
        barrier()

    def reload_states(self, model, optimizer=None) -> None:
        """Reload states to GPU."""
        device = torch.device("cuda", torch.cuda.current_device())
        unwrap_actor(model).to(device)
        if optimizer:
            move_optimizer_state(optimizer, device)
        torch.cuda.synchronize()
        barrier()

    # =========================================================================
    # Checkpointing
    # =========================================================================

    def save_model(self, model: nn.Module, tokenizer, output_dir: str, **kwargs) -> None:
        """Save model to HuggingFace format."""
        save_hf_model(
            model, tokenizer, output_dir, self.is_rank_0(), unwrap_actor, get_runtime_metadata(self), **kwargs
        )

    def load_model(self, model: nn.Module, path: str, **kwargs) -> None:
        """Load model weights."""
        load_hf_model(model, path, unwrap_actor, **kwargs)

    def save_ckpt(self, model: nn.Module, save_dir: str, tag: Optional[str] = None, **kwargs) -> None:
        """Save distributed checkpoint."""
        save_distributed_checkpoint(model, save_dir, tag, unwrap_actor, self.is_rank_0(), **kwargs)

    def load_ckpt(self, model: nn.Module, load_dir: str, **kwargs) -> Tuple[str, dict]:
        """Load distributed checkpoint."""
        return load_distributed_checkpoint(model, load_dir, unwrap_actor, **kwargs)

    # =========================================================================
    # Communication
    # =========================================================================

    def all_reduce(self, data, op: str = "mean"):
        """All-reduce across DP group."""
        if isinstance(data, dict):
            return {k: self.all_reduce(v, op) for k, v in data.items()}

        dp_group = self._get_dp_group()
        dp_size = dist.get_world_size(dp_group) if dist.is_initialized() else 1

        is_tensor = isinstance(data, torch.Tensor)
        tensor = data if is_tensor else torch.tensor(data)
        is_cpu = tensor.device.type == "cpu"
        if is_cpu:
            tensor = tensor.cuda()

        ops = {"mean": dist.ReduceOp.SUM, "max": dist.ReduceOp.MAX, "sum": dist.ReduceOp.SUM}
        dist.all_reduce(tensor, op=ops[op], group=dp_group)
        if op == "mean":
            tensor = tensor / dp_size

        if is_cpu:
            tensor = tensor.cpu()
        return tensor if is_tensor else tensor.item()

    def all_gather(self, data):
        """All-gather across DP group."""
        dp_group = self._get_dp_group()
        dp_size = dist.get_world_size(dp_group) if dist.is_initialized() else 1

        is_tensor = isinstance(data, torch.Tensor)
        tensor = data if is_tensor else torch.tensor(data)
        is_cpu = tensor.device.type == "cpu"

        outputs = [torch.zeros_like(tensor).cuda() for _ in range(dp_size)]
        dist.all_gather(outputs, tensor.cuda(), group=dp_group)
        result = torch.cat(outputs)

        return (result.cpu() if is_cpu else result) if is_tensor else result.tolist()

    # =========================================================================
    # Utilities
    # =========================================================================

    def _log(self, msg: str) -> None:
        """Log message on rank 0."""
        if self.is_rank_0():
            print(f"[FSDP2] {msg}")

    def print(self, *msg) -> None:
        """Print on rank 0 (legacy interface)."""
        if self.is_rank_0():
            print(*msg)

    def is_rank_0(self) -> bool:
        return not dist.is_initialized() or dist.get_rank() == 0

    def get_rank(self) -> int:
        return dist.get_rank() if dist.is_initialized() else 0

    def get_world_size(self) -> int:
        return dist.get_world_size() if dist.is_initialized() else 1

    def get_ds_train_config(self, is_actor: bool = False):
        """DeepSpeed compatibility - returns None for FSDP2."""
        return None

    def get_ds_eval_config(self, offload: bool = False):
        """DeepSpeed compatibility - returns None for FSDP2."""
        return None
