"""FSDP2 Strategy for OpenRLHF.

This module provides an FSDP2-based training strategy that mirrors the DeepSpeed interface.
It uses PyTorch's FSDP2 (fully_shard) for distributed training.

Key features:
- Automatic model sharding across GPUs
- Mixed precision training with configurable dtypes
- Tensor parallelism support via HuggingFace's ._tp_plan (AutoTP)
- Ring attention for sequence parallelism
- Standalone implementation (no abstraction layer)
"""

import os
import shutil
from abc import ABC
from collections import defaultdict
from datetime import timedelta
from typing import Any, List, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
import transformers
import transformers.modeling_flash_attention_utils
from torch import distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    OffloadPolicy,
    fully_shard,
)
from torch.distributed.tensor.parallel import parallelize_module
from torch.optim import Optimizer
from torchdata.stateful_dataloader import StatefulDataLoader

from openrlhf.models import Actor
from openrlhf.models.ring_attn_utils import get_ring_attn_group, set_ring_attn_group
from openrlhf.utils.distributed_sampler import DistributedSampler
from openrlhf.utils.distributed_util import torch_dist_barrier_and_cuda_sync

from .fsdp2_utils import (
    clip_grad_by_total_norm_,
    get_grad_norm,
    get_hf_tp_plan,
    get_llama_tp_plan,
    get_optimized_tp_plan,
    get_optimizer_grouped_parameters,
    to_local_if_dtensor,
)

ModelOptimPair = Tuple[nn.Module, Optimizer]
ModelOrModelOptimPair = Union[nn.Module, ModelOptimPair]


class FSDP2Strategy(ABC):
    """
    FSDP2-based training strategy for OpenRLHF.

    This strategy uses PyTorch FSDP2 (fully_shard) instead of DeepSpeed for distributed training.
    It provides the same interface as DeepspeedStrategy for compatibility.

    Features:
    - Automatic model sharding via FSDP2
    - Mixed precision training (bf16/fp16)
    - Tensor parallelism via HF's ._tp_plan
    - Ring attention for long sequences
    """

    def __init__(
        self,
        seed: int = 42,
        full_determinism: bool = False,
        max_norm: float = 1.0,
        micro_train_batch_size: int = 1,
        train_batch_size: int = 1,
        zero_stage: int = 2,  # Unused but kept for compatibility
        args=None,
    ) -> None:
        super().__init__()

        # Base attributes (same as DeepspeedStrategy)
        self.args = args
        self.stage = zero_stage
        self.train_batch_size = train_batch_size
        self.micro_train_batch_size = micro_train_batch_size
        self.seed = seed
        self.full_determinism = full_determinism
        self.max_norm = max_norm

        # Will be set in setup_distributed
        self.world_size = 1
        self.dp_size = 1
        self.accumulated_gradient = 1

        # Tracking for EMA updates
        self.time_steps = defaultdict(int)

        # Ring attention tracking
        self.ring_attn_rank = 0

        # RLHF mode flag
        self.is_rlhf = False

        self.param_dtype = args.param_dtype  # default: bf16

        # FSDP2-specific settings
        self.cpu_offload = getattr(args, "adam_offload", False)
        self.activation_checkpointing = getattr(args, "gradient_checkpointing", False)
        self.tensor_parallel_size = getattr(args, "fsdp_tensor_parallel_size", 1)
        self.ring_attn_size = getattr(args, "ring_attn_size", 1)
        self.use_dynamic_batch = getattr(args, "use_dynamic_batch", False)

        # AutoTP settings (HuggingFace's built-in tensor parallel plan)
        self.use_hf_tp_plan = getattr(args, "use_hf_tp_plan", False)
        self.sequence_parallel = getattr(args, "sequence_parallel", False)

        # Validate sequence parallel configuration
        if self.sequence_parallel and self.tensor_parallel_size == 1:
            print(
                "[Warning] sequence_parallel=True but tensor_parallel_size=1, which has no effect. "
                "Enable tensor_parallel_size > 1 to use sequence parallelism."
            )

        # Will be set in setup_distributed
        self.device_mesh = None
        self.dp_mesh = None
        self.tp_mesh = None

        # Gradient accumulation tracking
        self._step_count = 0

    def setup_distributed(self, timeout=timedelta(minutes=60)) -> None:
        """Initialize distributed training environment."""
        if self.full_determinism:
            transformers.enable_full_determinism(self.seed)
            transformers.modeling_flash_attention_utils.deterministic_g = True
        else:
            transformers.set_seed(self.seed)

        # Take the local rank from args as first priority
        if self.args.local_rank != -1:
            os.environ["LOCAL_RANK"] = str(self.args.local_rank)

        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        if local_rank != -1:
            torch.cuda.set_device(local_rank)

        # Initialize process group
        if not dist.is_initialized():
            backend = "nccl"
            dist.init_process_group(backend=backend, timeout=timeout)

        self.world_size = dist.get_world_size()

        # Validate parallelism configuration
        parallel_product = self.ring_attn_size * self.tensor_parallel_size
        if self.world_size % parallel_product != 0:
            raise ValueError(
                f"world_size ({self.world_size}) must be divisible by "
                f"ring_attn_size ({self.ring_attn_size}) * tensor_parallel_size ({self.tensor_parallel_size}) = {parallel_product}"
            )

        self.dp_size = self.world_size // self.ring_attn_size // self.tensor_parallel_size

        if self.dp_size < 1:
            raise ValueError(
                f"Data parallel size must be >= 1, got {self.dp_size}. "
                f"Reduce ring_attn_size ({self.ring_attn_size}) or tensor_parallel_size ({self.tensor_parallel_size})"
            )

        # Print configuration on rank 0
        if dist.get_rank() == 0:
            print(f"[FSDP2] Distributed configuration:")
            print(f"  - World size: {self.world_size}")
            print(f"  - Data parallel size: {self.dp_size}")
            print(f"  - Tensor parallel size: {self.tensor_parallel_size}")
            print(f"  - Ring attention size: {self.ring_attn_size}")

        # Create device mesh
        self.device_mesh = init_device_mesh(
            "cuda", (self.dp_size, self.ring_attn_size, self.tensor_parallel_size), mesh_dim_names=("dp", "sp", "tp")
        )

        self.dp_mesh = self.device_mesh["dp"]
        if self.tensor_parallel_size > 1:
            self.tp_mesh = self.device_mesh["tp"]
        else:
            self.tp_mesh = None

        self.setup_ring_attn(self.device_mesh)

        self.accumulated_gradient = (
            self.train_batch_size
            * self.ring_attn_size
            * self.tensor_parallel_size
            // self.micro_train_batch_size
            // self.world_size
        )

        if self.accumulated_gradient < 1:
            raise ValueError(
                f"Gradient accumulation steps must be >= 1, got {self.accumulated_gradient}. "
                f"Increase train_batch_size ({self.train_batch_size}) or decrease micro_train_batch_size ({self.micro_train_batch_size})"
            )

        if dist.get_rank() == 0:
            print(f"  - Gradient accumulation steps: {self.accumulated_gradient}")

    def setup_ring_attn(self, device_mesh):
        """Setup ring attention if enabled.

        This overrides the base class method to handle FSDP2-specific setup.
        The ring attention group is used for distributing attention computation
        across multiple GPUs in a ring topology.

        Args:
            device_mesh: The device mesh containing the 'sp' (sequence parallel) dimension
        """
        if self.ring_attn_size == 1:
            self.ring_attn_rank = 0
            return

        # Get the ring attention group from the device mesh
        group = device_mesh["sp"].get_group()
        self.ring_attn_rank = dist.get_rank(group=group)

        # Set the global ring attention group
        set_ring_attn_group(group)

        # Substitute HuggingFace flash attention with ring flash attention
        from ring_flash_attn import substitute_hf_flash_attn

        self.ring_head_stride = getattr(self.args, "ring_head_stride", 1)
        # Use the group directly instead of self.ring_attn_group property
        # to avoid the property lookup before the group is fully set
        substitute_hf_flash_attn(group, self.ring_head_stride)

        if self.is_rank_0():
            print(f"[FSDP2] Ring attention enabled with size {self.ring_attn_size}")

    @property
    def ring_attn_group(self):
        """Get the ring attention process group."""
        return get_ring_attn_group()

    def create_optimizer(self, model, **kwargs) -> Optimizer:
        """Create optimizer for the model."""
        if isinstance(model, Actor):
            model = model.model

        optim_params = get_optimizer_grouped_parameters(model, kwargs.get("weight_decay", 0.0))

        # Use FusedAdam if available, otherwise fallback to AdamW
        try:
            from apex.optimizers import FusedAdam

            optimizer = FusedAdam(optim_params, **kwargs)
        except (ImportError, RuntimeError):
            # Fallback to AdamW if FusedAdam is not available or CUDA extensions not built
            optimizer = torch.optim.AdamW(optim_params, **kwargs)

        return optimizer

    def backward(self, loss: torch.Tensor, model: nn.Module, optimizer: optim.Optimizer, **kwargs) -> None:
        """Perform backward pass with gradient accumulation handling.

        For FSDP2:
        - Scale loss for gradient accumulation
        - FSDP2 automatically handles gradient averaging across data parallel ranks
        """
        # Scale loss for gradient accumulation
        # The gradients are accumulated across micro-batches, and we take the average
        scaled_loss = loss / self.accumulated_gradient
        scaled_loss.backward()

        self._step_count += 1

    def optimizer_step(
        self,
        optimizer: optim.Optimizer,
        model: nn.Module,
        scheduler,
        name="model",
        **kwargs,
    ) -> None:
        """Perform optimizer step with gradient clipping.

        This is called after each micro-batch backward pass.
        We only actually step the optimizer after accumulating enough gradients.
        """
        if isinstance(model, Actor):
            model = model.model

        # Only step when we've accumulated enough gradients
        if self._step_count % self.accumulated_gradient != 0:
            return

        # Gradient clipping
        if self.max_norm > 0:
            with torch.no_grad():
                total_norm = get_grad_norm(
                    list(model.parameters()),
                    dp_group=self.dp_mesh.get_group(),
                    tp_group=self.tp_mesh.get_group() if self.tp_mesh is not None else None,
                    dtype=torch.float32,
                )
                clip_grad_by_total_norm_(
                    list(model.parameters()),
                    max_grad_norm=self.max_norm,
                    total_norm=total_norm,
                )

        optimizer.step()
        optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

    def setup_dataloader(
        self,
        replay_buffer,
        batch_size: int,
        pin_memory: bool = False,
        shuffle=True,
        collate_fn=None,
        drop_last=True,
        sampler=None,
        consumed_samples=0,
    ):
        """Setup dataloader with distributed sampler."""
        if sampler is None and dist.is_initialized():
            dp_group = self.dp_mesh.get_group()
            num_replicas = dist.get_world_size(group=dp_group)
            rank = dist.get_rank(group=dp_group)

            sampler = DistributedSampler(
                replay_buffer,
                num_replicas=num_replicas,
                rank=rank,
                shuffle=shuffle,
                seed=self.seed,
                drop_last=drop_last,
                consumed_samples=consumed_samples,
            )

        return StatefulDataLoader(
            replay_buffer,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=drop_last,
            shuffle=shuffle if sampler is None else False,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
        )

    def _unwrap_model(self, model) -> nn.Module:
        """Unwrap model from Actor wrapper."""
        if isinstance(model, Actor):
            return self._unwrap_model(model.model)
        return model

    def prepare(
        self, *models_or_model_optim_pairs: ModelOrModelOptimPair, is_rlhf=False
    ) -> Union[List[ModelOrModelOptimPair], ModelOrModelOptimPair]:
        """Prepare models for distributed training with FSDP2."""
        ret = []
        self.is_rlhf = is_rlhf

        for arg in models_or_model_optim_pairs:
            if isinstance(arg, tuple):
                assert len(arg) == 3, f'Expect (model, optimizer, scheduler) pair, got a tuple with size "{len(arg)}"'
                if arg[0] is not None:
                    ret.append(self._fsdp2_init_train_model(*arg))
                else:
                    ret.append((None, None, None))
            else:
                ret.append(self._fsdp2_init_eval_model(arg))

        return ret[0] if len(ret) == 1 else ret

    def _recreate_scheduler(self, scheduler: Any, new_optimizer: Optimizer) -> Any:
        """Recreate a learning rate scheduler for the new optimizer.

        When FSDP2 wraps the model, the original optimizer is replaced. This method
        recreates the scheduler to work with the new optimizer while preserving
        the original scheduler's configuration and state.

        Args:
            scheduler: The original scheduler to recreate
            new_optimizer: The new optimizer to use with the scheduler

        Returns:
            New scheduler instance configured like the original
        """
        scheduler_state = scheduler.state_dict()
        scheduler_type = type(scheduler)
        new_scheduler = None

        try:
            # Check for LambdaLR scheduler (most common in transformers)
            # This includes schedulers from transformers.get_scheduler()
            if hasattr(scheduler, "lr_lambdas") and scheduler.lr_lambdas is not None:
                # LambdaLR or similar - preserve the lr_lambda functions
                new_scheduler = torch.optim.lr_scheduler.LambdaLR(new_optimizer, lr_lambda=scheduler.lr_lambdas)
            # ChainedScheduler (used by transformers for warmup + decay)
            elif hasattr(scheduler, "_schedulers"):
                # Recursively recreate chained schedulers
                new_schedulers = []
                for s in scheduler._schedulers:
                    new_schedulers.append(self._recreate_scheduler(s, new_optimizer))
                new_scheduler = torch.optim.lr_scheduler.ChainedScheduler(new_schedulers)
            # SequentialLR
            elif hasattr(scheduler, "_schedulers") and hasattr(scheduler, "_milestones"):
                new_schedulers = []
                for s in scheduler._schedulers:
                    new_schedulers.append(self._recreate_scheduler(s, new_optimizer))
                new_scheduler = torch.optim.lr_scheduler.SequentialLR(
                    new_optimizer, schedulers=new_schedulers, milestones=scheduler._milestones
                )
            # CosineAnnealingLR
            elif hasattr(scheduler, "T_max"):
                new_scheduler = scheduler_type(
                    new_optimizer,
                    T_max=scheduler.T_max,
                    eta_min=getattr(scheduler, "eta_min", 0),
                )
            # CosineAnnealingWarmRestarts
            elif hasattr(scheduler, "T_0"):
                new_scheduler = scheduler_type(
                    new_optimizer,
                    T_0=scheduler.T_0,
                    T_mult=getattr(scheduler, "T_mult", 1),
                    eta_min=getattr(scheduler, "eta_min", 0),
                )
            # LinearLR
            elif hasattr(scheduler, "total_iters") and hasattr(scheduler, "start_factor"):
                new_scheduler = scheduler_type(
                    new_optimizer,
                    start_factor=scheduler.start_factor,
                    end_factor=getattr(scheduler, "end_factor", 1.0),
                    total_iters=scheduler.total_iters,
                )
            # ConstantLR
            elif hasattr(scheduler, "total_iters") and hasattr(scheduler, "factor"):
                new_scheduler = scheduler_type(
                    new_optimizer,
                    factor=scheduler.factor,
                    total_iters=scheduler.total_iters,
                )
            # ExponentialLR
            elif (
                hasattr(scheduler, "gamma")
                and not hasattr(scheduler, "step_size")
                and not hasattr(scheduler, "milestones")
            ):
                new_scheduler = scheduler_type(
                    new_optimizer,
                    gamma=scheduler.gamma,
                )
            # MultiStepLR
            elif hasattr(scheduler, "milestones") and hasattr(scheduler, "gamma"):
                new_scheduler = scheduler_type(
                    new_optimizer,
                    milestones=list(scheduler.milestones),
                    gamma=scheduler.gamma,
                )
            # StepLR
            elif hasattr(scheduler, "step_size") and hasattr(scheduler, "gamma"):
                new_scheduler = scheduler_type(
                    new_optimizer,
                    step_size=scheduler.step_size,
                    gamma=scheduler.gamma,
                )
            # ReduceLROnPlateau (special case - doesn't have load_state_dict compatibility)
            elif scheduler_type.__name__ == "ReduceLROnPlateau":
                new_scheduler = scheduler_type(
                    new_optimizer,
                    mode=scheduler.mode,
                    factor=scheduler.factor,
                    patience=scheduler.patience,
                    threshold=scheduler.threshold,
                    threshold_mode=scheduler.threshold_mode,
                    cooldown=scheduler.cooldown,
                    min_lr=scheduler.min_lrs[0] if scheduler.min_lrs else 0,
                    eps=scheduler.eps,
                )
                # ReduceLROnPlateau state is handled separately
                return new_scheduler
            # Fallback: try to recreate using generic approach
            else:
                self.print(
                    f"[FSDP2] Warning: Unknown scheduler type {scheduler_type.__name__}, "
                    "attempting generic recreation"
                )
                try:
                    new_scheduler = scheduler_type(new_optimizer)
                except TypeError:
                    # If generic recreation fails, use constant LR as ultimate fallback
                    self.print(f"[FSDP2] Warning: Could not recreate scheduler, using constant LR")
                    new_scheduler = torch.optim.lr_scheduler.LambdaLR(new_optimizer, lr_lambda=lambda epoch: 1)
                    return new_scheduler

            # Load the old scheduler state (adjusts last_epoch, etc.)
            try:
                new_scheduler.load_state_dict(scheduler_state)
            except Exception as e:
                self.print(f"[FSDP2] Warning: Could not load scheduler state: {e}")

        except Exception as e:
            self.print(f"[FSDP2] Warning: Error recreating scheduler: {e}")
            self.print(f"[FSDP2] Falling back to constant LR scheduler")
            new_scheduler = torch.optim.lr_scheduler.LambdaLR(new_optimizer, lr_lambda=lambda epoch: 1)

        return new_scheduler

    def _get_tp_plan(self, model: nn.Module):
        """Get tensor parallel plan with fallback priority.

        The fallback priority is:
        1. HuggingFace's built-in TP plan (if use_hf_tp_plan=True and available)
        2. Optimized TP plan for known model architectures
        3. Default LLaMA-style TP plan

        Args:
            model: The model to get the TP plan for

        Returns:
            Dictionary mapping module paths to parallel styles
        """
        tp_plan = None
        plan_source = "default"

        # Try HF tp_plan first if enabled
        if self.use_hf_tp_plan:
            try:
                tp_plan = get_hf_tp_plan(model)
                plan_source = "HuggingFace built-in"
                self.print(f"[FSDP2] Using HuggingFace's built-in tensor parallel plan")
            except AssertionError as e:
                self.print(f"[FSDP2] HuggingFace tp_plan not available: {e}")
                self.print("[FSDP2] Falling back to optimized/default plan")

        # Try optimized plan if HF plan not available
        if tp_plan is None:
            tp_plan = get_optimized_tp_plan(model, sequence_parallel=self.sequence_parallel)
            if tp_plan is not None:
                plan_source = "optimized"
                self.print(f"[FSDP2] Using optimized tensor parallel plan for {type(model).__name__}")

        # Fall back to default LLaMA-style plan
        if tp_plan is None:
            tp_plan = get_llama_tp_plan(sequence_parallel=self.sequence_parallel)
            plan_source = "default (LLaMA-style)"
            self.print(f"[FSDP2] Using default LLaMA-style tensor parallel plan")

        return tp_plan

    def _fsdp2_init_train_model(self, model, optim, scheduler):
        """Initialize model with FSDP2 for training.

        Note: The optimizer needs to be recreated after FSDP2 wrapping because
        FSDP2 replaces model parameters with DTensors. The scheduler also needs
        to be recreated since it references the optimizer.
        """
        is_actor = isinstance(model, Actor)
        inner_model = model.model if is_actor else model

        # Get dtype
        if self.param_dtype == "bf16":
            dtype = torch.bfloat16
        elif self.param_dtype == "fp16":
            dtype = torch.float16
        else:
            dtype = torch.float32

        # Apply tensor parallelism if enabled
        if self.tensor_parallel_size > 1 and self.tp_mesh is not None:
            # Validate that num_attention_heads and num_key_value_heads are divisible by TP size
            config = inner_model.config if hasattr(inner_model, "config") else None
            if config is not None:
                num_attention_heads = getattr(config, "num_attention_heads", None)
                num_key_value_heads = getattr(config, "num_key_value_heads", num_attention_heads)

                if num_attention_heads is not None:
                    assert num_attention_heads % self.tensor_parallel_size == 0, (
                        f"num_attention_heads ({num_attention_heads}) must be divisible by "
                        f"tensor_parallel_size ({self.tensor_parallel_size})"
                    )
                if num_key_value_heads is not None:
                    assert num_key_value_heads % self.tensor_parallel_size == 0, (
                        f"num_key_value_heads ({num_key_value_heads}) must be divisible by "
                        f"tensor_parallel_size ({self.tensor_parallel_size})"
                    )

            tp_plan = self._get_tp_plan(inner_model)
            parallelize_module(inner_model, self.tp_mesh, tp_plan)

        # Apply FSDP2
        inner_model = self._apply_fsdp2(inner_model, dtype, is_training=True)

        if is_actor:
            model.model = inner_model
        else:
            model = inner_model

        # Recreate optimizer with FSDP-wrapped model parameters
        # This is necessary because FSDP replaces parameters with DTensors
        new_optim = None
        new_scheduler = None

        if optim is not None:
            # Get optimizer config from the old optimizer
            old_lr = optim.param_groups[0]["lr"]
            old_betas = optim.param_groups[0].get("betas", (0.9, 0.999))
            old_weight_decay = optim.param_groups[0].get("weight_decay", 0.0)

            # Create new optimizer with FSDP-wrapped parameters
            optim_params = get_optimizer_grouped_parameters(inner_model, old_weight_decay)

            try:
                from apex.optimizers import FusedAdam

                new_optim = FusedAdam(optim_params, lr=old_lr, betas=old_betas)
            except (ImportError, RuntimeError):
                # Fallback to AdamW if FusedAdam is not available or CUDA extensions not built
                new_optim = torch.optim.AdamW(optim_params, lr=old_lr, betas=old_betas)

            # Recreate scheduler if it exists
            if scheduler is not None:
                new_scheduler = self._recreate_scheduler(scheduler, new_optim)

        return (
            model,
            new_optim if new_optim is not None else optim,
            new_scheduler if new_scheduler is not None else scheduler,
        )

    def _fsdp2_init_eval_model(self, model):
        """Initialize model with FSDP2 for evaluation."""
        if model is None:
            return model

        is_actor = isinstance(model, Actor)
        inner_model = model.model if is_actor else model

        # Get dtype
        if self.param_dtype == "bf16":
            dtype = torch.bfloat16
        elif self.param_dtype == "fp16":
            dtype = torch.float16
        else:
            dtype = torch.float32

        # Apply tensor parallelism if enabled
        if self.tensor_parallel_size > 1 and self.tp_mesh is not None:
            tp_plan = self._get_tp_plan(inner_model)
            parallelize_module(inner_model, self.tp_mesh, tp_plan)

        # Apply FSDP2
        inner_model = self._apply_fsdp2(inner_model, dtype, is_training=False)

        if is_actor:
            model.model = inner_model
        else:
            model = inner_model

        return model

    def _apply_fsdp2(self, model: nn.Module, dtype: torch.dtype, is_training: bool = True):
        """Apply FSDP2 to the model.

        Args:
            model: The model to wrap with FSDP2
            dtype: The parameter dtype for mixed precision
            is_training: Whether this is for training or evaluation

        Returns:
            The FSDP2-wrapped model
        """
        # Mixed precision policy
        # Note: When gradient checkpointing is enabled, we keep output_dtype same as param_dtype
        # to avoid dtype mismatch during gradient checkpointing recomputation.
        # The final logits are cast to float32 in Actor.forward() for loss computation.
        output_dtype = dtype if self.activation_checkpointing else torch.float32
        mp_policy = MixedPrecisionPolicy(
            param_dtype=dtype,
            reduce_dtype=torch.float32,
            output_dtype=output_dtype,
        )

        # Offload policy
        offload_policy = CPUOffloadPolicy(pin_memory=False) if self.cpu_offload else OffloadPolicy()

        # Get transformer layers to shard
        layers = None
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
        elif hasattr(model, "layers"):
            layers = model.layers

        # Apply FSDP2 to each transformer layer
        if layers is not None:
            for layer in layers:
                fully_shard(
                    layer,
                    mesh=self.dp_mesh,
                    mp_policy=mp_policy,
                    offload_policy=offload_policy,
                )

        # Apply FSDP2 to the whole model (don't reshard after forward for root)
        model = fully_shard(
            model,
            mesh=self.dp_mesh,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
            reshard_after_forward=False,
        )

        return model

    def get_ds_train_config(self, is_actor):
        """Get dummy config for compatibility with Actor init.

        This returns a minimal dict since FSDP2 doesn't use DeepSpeed config.
        """
        return {
            "zero_optimization": {"stage": 0},  # Minimal config for compatibility
        }

    def get_ds_eval_config(self, offload=False):
        """Get dummy config for compatibility."""
        return {
            "zero_optimization": {"stage": 0},
        }

    def moving_average(self, model, model_ema, beta=0.992, device="cpu"):
        """Update EMA model weights with DTensor support.

        For FSDP2, parameters may be DTensors, so we need to convert them to
        local tensors before performing the EMA update.

        Args:
            model: Source model
            model_ema: EMA model to update
            beta: EMA coefficient (default: 0.992)
            device: Device for computation (default: 'cpu')
        """
        self.time_steps["ema"] += 1
        if self.time_steps["ema"] % self.accumulated_gradient == 0 or self.use_dynamic_batch:
            with torch.no_grad():
                for param, param_ema in zip(model.parameters(), model_ema.parameters()):
                    if param.requires_grad:
                        # Convert DTensor to local tensor if needed
                        data = to_local_if_dtensor(param.data).to(device)
                        param_ema_data = to_local_if_dtensor(param_ema.data)
                        param_ema_data.copy_((1 - beta) * data + beta * param_ema_data)

    def load_model(
        self,
        model: nn.Module,
        path: str,
        map_location="cpu",
        strict: bool = False,
        key_replace_fn=None,
    ) -> None:
        """Load model weights from a checkpoint."""
        unwrapped_model = self._unwrap_model(model)
        state_dict = torch.load(path, map_location=map_location)
        if key_replace_fn:
            state_dict = key_replace_fn(state_dict)
        unwrapped_model.load_state_dict(state_dict, strict=strict)

    def save_model(self, model: nn.Module, tokenizer, output_dir, **kwargs) -> None:
        """Save model weights in HuggingFace format.

        For FSDP2, we use torch.distributed.checkpoint utilities to gather
        the full state dict from all ranks.
        """
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

        if self.is_rank_0():
            os.makedirs(output_dir, exist_ok=True)

        model_to_save = self._unwrap_model(model)

        # Gather full state dict from FSDP shards using torch.distributed.checkpoint
        with torch.no_grad():
            # Get the full state dict - this gathers from all ranks
            state_dict = get_model_state_dict(
                model_to_save,
                options=StateDictOptions(
                    full_state_dict=True,
                    cpu_offload=True,
                ),
            )

        if self.is_rank_0():
            # Save model
            model_to_save.save_pretrained(output_dir, state_dict=state_dict, **kwargs)

            # Save config
            output_config_file = os.path.join(output_dir, "config.json")
            model_to_save.config.to_json_file(output_config_file)

            # Save tokenizer
            tokenizer.save_pretrained(output_dir)

        del state_dict
        import gc

        gc.collect()

        torch_dist_barrier_and_cuda_sync()

    def all_reduce(self, data, op="mean"):
        """All-reduce data across all ranks."""
        assert op in ("mean", "max", "sum")
        if isinstance(data, dict):
            ret = {}
            for k, v in data.items():
                ret[k] = self.all_reduce(v, op)
            return ret
        else:
            is_tensor = True
            if not isinstance(data, torch.Tensor):
                data = torch.Tensor([data])
                is_tensor = False
            is_cpu_tensor = data.device.type == "cpu"

            if is_cpu_tensor:
                data = data.to(torch.cuda.current_device())
            if op == "mean":
                data /= self.world_size
            dist.all_reduce(data, op=dist.ReduceOp.MAX if op == "max" else dist.ReduceOp.SUM)
            if is_cpu_tensor:
                data = data.cpu()
            return data.item() if not is_tensor else data

    def all_gather(self, data):
        """All-gather data from all ranks."""
        if isinstance(data, dict):
            ret = {}
            for k, v in data.items():
                ret[k] = self.all_gather(v)
            return ret
        else:
            if not isinstance(data, torch.Tensor):
                data = torch.Tensor([data])
            is_cpu_tensor = data.device.type == "cpu"

            ret = [torch.zeros_like(data).to(torch.cuda.current_device()) for _ in range(self.world_size)]
            dist.all_gather(ret, data.to(torch.cuda.current_device()))
            return torch.cat(ret).cpu() if is_cpu_tensor else torch.cat(ret)

    def print(self, *msg):
        """Print message only on rank 0."""
        if self.is_rank_0():
            print(*msg)

    def is_rank_0(self) -> bool:
        """Check if current rank is rank 0."""
        if not dist.is_initialized():
            return True
        return dist.get_rank() == 0

    def get_rank(self) -> int:
        """Get current rank."""
        if not dist.is_initialized():
            return 0
        return dist.get_rank()

    def save_ckpt(self, model, save_dir, tag=None, max_num=3, max_mem=1000, client_state={}, save_latest=True):
        """Save checkpoint for training resumption.

        For FSDP2, we save using torch.distributed.checkpoint for proper sharded saving.
        """
        from torch.distributed.checkpoint import save
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

        if self.is_rank_0():
            os.makedirs(save_dir, exist_ok=True)
            MAX_SIZE = max_mem * 1024**3

            # Clean old checkpoints
            while True:
                subdirs = sorted(
                    [
                        (os.path.join(save_dir, d), os.path.getmtime(os.path.join(save_dir, d)))
                        for d in os.listdir(save_dir)
                        if os.path.isdir(os.path.join(save_dir, d))
                    ],
                    key=lambda x: x[1],
                )
                total_size = sum(
                    os.path.getsize(os.path.join(dirpath, f))
                    for subdir, _ in subdirs
                    for dirpath, _, filenames in os.walk(subdir)
                    for f in filenames
                )

                if len(subdirs) >= max_num or total_size > MAX_SIZE:
                    oldest_dir = subdirs[0][0]
                    if os.path.exists(oldest_dir):
                        shutil.rmtree(oldest_dir)
                        self.print(f"Deleted oldest ckpt {oldest_dir}")
                else:
                    break

        torch_dist_barrier_and_cuda_sync()

        # Save checkpoint
        ckpt_dir = os.path.join(save_dir, tag if tag else "latest")

        # Get state dict
        state_dict = {
            "model": get_model_state_dict(model, options=StateDictOptions(full_state_dict=False)),
            "client_state": client_state,
        }

        save(state_dict, checkpoint_id=ckpt_dir)

        import gc

        gc.collect()

    def load_ckpt(
        self,
        model,
        load_dir,
        tag=None,
        load_module_strict=True,
        load_optimizer_states=True,
        load_lr_scheduler_states=True,
        load_module_only=False,
    ):
        """Load checkpoint for training resumption."""
        from torch.distributed.checkpoint import load
        from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

        ckpt_dir = os.path.join(load_dir, tag if tag else "latest")

        if not os.path.exists(ckpt_dir):
            raise Exception(f"[FSDP2] failed to resume from checkpoint {ckpt_dir}")

        state_dict = {"model": {}, "client_state": {}}
        load(state_dict, checkpoint_id=ckpt_dir)

        set_model_state_dict(
            model,
            model_state_dict=state_dict["model"],
            options=StateDictOptions(strict=load_module_strict),
        )

        return ckpt_dir, state_dict.get("client_state", {})
