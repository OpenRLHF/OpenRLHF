"""FSDP2 Strategy for OpenRLHF.

This module provides an FSDP2-based training strategy that mirrors the DeepSpeed interface.
It uses PyTorch's FSDP2 (fully_shard) for distributed training.
"""

import os
import shutil
from abc import ABC
from collections import defaultdict
from datetime import timedelta
from typing import List, Tuple, Union, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import transformers
import transformers.modeling_flash_attention_utils
from torch import distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    fully_shard,
    MixedPrecisionPolicy,
    CPUOffloadPolicy,
    OffloadPolicy,
    FSDPModule,
)
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import parallelize_module
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.optim import Optimizer
from torchdata.stateful_dataloader import StatefulDataLoader

from openrlhf.models import Actor
from openrlhf.models.ring_attn_utils import get_ring_attn_group, set_ring_attn_group
from openrlhf.utils.distributed_sampler import DistributedSampler
from openrlhf.utils.distributed_util import torch_dist_barrier_and_cuda_sync
from .fsdp2_utils import (
    get_optimizer_grouped_parameters,
    get_grad_norm,
    clip_grad_by_total_norm_,
    to_local_if_dtensor,
    get_llama_tp_plan,
)

ModelOptimPair = Tuple[nn.Module, Optimizer]
ModelOrModelOptimPair = Union[nn.Module, ModelOptimPair]


class FSDP2Strategy(ABC):
    """
    FSDP2-based training strategy for OpenRLHF.
    
    This strategy uses PyTorch FSDP2 (fully_shard) instead of DeepSpeed for distributed training.
    It provides the same interface as DeepspeedStrategy for compatibility.
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

        self.args = args
        self.stage = zero_stage  # Kept for compatibility
        self.train_batch_size = train_batch_size
        self.micro_train_batch_size = micro_train_batch_size
        self.param_dtype = args.param_dtype  # default: bf16
        self.seed = seed
        self.full_determinism = full_determinism
        self.max_norm = max_norm

        # FSDP2-specific settings
        self.cpu_offload = getattr(args, "adam_offload", False)
        self.activation_checkpointing = getattr(args, "gradient_checkpointing", False)
        self.tensor_parallel_size = getattr(args, "fsdp_tensor_parallel_size", 1)
        self.ring_attn_size = getattr(args, "ring_attn_size", 1)
        self.use_dynamic_batch = getattr(args, "use_dynamic_batch", False)

        self.is_rlhf = False
        self.time_steps = defaultdict(int)
        
        # Will be set in setup_distributed
        self.world_size = 1
        self.dp_size = 1
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
        self.dp_size = self.world_size // self.ring_attn_size // self.tensor_parallel_size

        # Create device mesh
        self.device_mesh = init_device_mesh(
            "cuda",
            (self.dp_size, self.ring_attn_size, self.tensor_parallel_size),
            mesh_dim_names=("dp", "sp", "tp")
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

    def setup_ring_attn(self, device_mesh):
        """Setup ring attention if enabled."""
        if self.ring_attn_size == 1:
            self.ring_attn_rank = 0
            return

        group = device_mesh["sp"].get_group()
        self.ring_attn_rank = dist.get_rank(group=group)
        set_ring_attn_group(group)

        from ring_flash_attn import substitute_hf_flash_attn

        self.ring_head_stride = getattr(self.args, "ring_head_stride", 1)
        substitute_hf_flash_attn(self.ring_attn_group, self.ring_head_stride)

    @property
    def ring_attn_group(self):
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
        except ImportError:
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
            tp_plan = get_llama_tp_plan(sequence_parallel=False)
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
            old_lr = optim.param_groups[0]['lr']
            old_betas = optim.param_groups[0].get('betas', (0.9, 0.999))
            old_weight_decay = optim.param_groups[0].get('weight_decay', 0.0)
            
            # Create new optimizer with FSDP-wrapped parameters
            optim_params = get_optimizer_grouped_parameters(inner_model, old_weight_decay)
            
            try:
                from apex.optimizers import FusedAdam
                new_optim = FusedAdam(optim_params, lr=old_lr, betas=old_betas)
            except ImportError:
                new_optim = torch.optim.AdamW(optim_params, lr=old_lr, betas=old_betas)
            
            # Recreate scheduler if it exists
            if scheduler is not None:
                # Try to get scheduler parameters
                # Most schedulers have these attributes
                scheduler_state = scheduler.state_dict()
                scheduler_type = type(scheduler)
                
                try:
                    # Try to recreate the same scheduler type
                    if hasattr(scheduler, 'T_max'):  # CosineAnnealingLR
                        new_scheduler = scheduler_type(
                            new_optim,
                            T_max=scheduler.T_max,
                            eta_min=getattr(scheduler, 'eta_min', 0),
                        )
                    elif hasattr(scheduler, 'total_iters'):  # LinearLR, etc.
                        new_scheduler = scheduler_type(
                            new_optim,
                            total_iters=scheduler.total_iters,
                        )
                    else:
                        # Fallback: create a simple LambdaLR that maintains constant LR
                        new_scheduler = torch.optim.lr_scheduler.LambdaLR(
                            new_optim, lr_lambda=lambda epoch: 1
                        )
                    
                    # Load the old scheduler state (adjusts last_epoch, etc.)
                    new_scheduler.load_state_dict(scheduler_state)
                except Exception as e:
                    self.print(f"Warning: Could not recreate scheduler, using constant LR: {e}")
                    new_scheduler = torch.optim.lr_scheduler.LambdaLR(
                        new_optim, lr_lambda=lambda epoch: 1
                    )

        return model, new_optim if new_optim is not None else optim, new_scheduler if new_scheduler is not None else scheduler

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
            tp_plan = get_llama_tp_plan(sequence_parallel=False)
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
        mp_policy = MixedPrecisionPolicy(
            param_dtype=dtype,
            reduce_dtype=torch.float32,
            output_dtype=torch.float32,
        )

        # Offload policy
        offload_policy = (
            CPUOffloadPolicy(pin_memory=False) if self.cpu_offload else OffloadPolicy()
        )

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
        """Update EMA model weights."""
        self.time_steps["ema"] += 1
        if self.time_steps["ema"] % self.accumulated_gradient == 0 or self.use_dynamic_batch:
            with torch.no_grad():
                for param, param_ema in zip(model.parameters(), model_ema.parameters()):
                    if param.requires_grad:
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
        from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
        
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
                )
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
        from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
        
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
        from torch.distributed.checkpoint.state_dict import set_model_state_dict, StateDictOptions
        
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
