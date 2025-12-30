import math
import os
from abc import ABC
from collections import defaultdict
from datetime import timedelta
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from packaging import version
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffloadPolicy, FSDPModule, MixedPrecisionPolicy, fully_shard
from torch.optim import Optimizer
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import enable_full_determinism, set_seed

from openrlhf.models import Actor
from openrlhf.models.ring_attn_utils import get_ring_attn_group, set_ring_attn_group
from openrlhf.utils import convert_to_dtype
from openrlhf.utils.distributed_sampler import DistributedSampler

ModelOptimPair = Tuple[nn.Module, Optimizer]
ModelOrModelOptimPair = Union[nn.Module, ModelOptimPair]


class FSDP2Strategy(ABC):
    """FSDP2 distributed training strategy.

    Device mesh layout: (dp, cp, tp)
    - dp: Data Parallel (FSDP sharding) - outer dimension
    - cp: Context Parallel (ring attention) - middle dimension
    - tp: Tensor Parallel - inner dimension (most contiguous)

    Supported parallelism combinations:
    - Pure DP: 1D mesh ("dp",)
    - DP + CP (ring attention): 2D mesh ("dp", "cp")
    - DP + TP: 2D mesh ("dp", "tp")
    - DP + CP + TP: 3D mesh ("dp", "cp", "tp")
    """

    def __init__(
        self,
        seed: int = 42,
        full_determinism: bool = False,
        max_norm: float = 0.0,
        micro_train_batch_size=1,
        train_batch_size=1,
        args=None,
    ) -> None:
        super().__init__()

        self.args = args
        self.train_batch_size = train_batch_size
        self.micro_train_batch_size = micro_train_batch_size
        self.seed = seed
        self.full_determinism = full_determinism
        self.max_norm = max_norm
        self.precision = getattr(args, "precision", "bf16")

        self.ring_attn_size = int(getattr(args, "ring_attn_size", 1) or 1)
        self.ds_tensor_parallel_size = int(getattr(args, "ds_tensor_parallel_size", 1) or 1)

        self.fsdp2_offload = getattr(args, "fsdp2_offload", "none")
        self.fsdp2_cpu_offload_pin_memory = getattr(args, "fsdp2_cpu_offload_pin_memory", True)
        self.fsdp2_reshard_after_forward = getattr(args, "fsdp2_reshard_after_forward", True)

        self.is_rlhf = False
        self.time_steps = defaultdict(int)

        torch_version = version.parse(torch.__version__.split("+")[0])
        required_version = version.parse("2.4.0")
        if torch_version < required_version:
            raise RuntimeError(
                "FSDP2 backend requires PyTorch >= 2.4 with the fully_shard API. "
                f"Detected torch=={torch.__version__}. Please upgrade PyTorch or use --dist_backend deepspeed."
            )

        self._offload_policy: Optional["CPUOffloadPolicy"] = self._build_offload_policy()
        self._mp_policy: Optional["MixedPrecisionPolicy"] = self._build_mixed_precision_policy()

    def _build_offload_policy(self) -> Optional["CPUOffloadPolicy"]:
        """Build CPU offload policy."""
        offload_mode = (self.fsdp2_offload or "none").lower()
        if offload_mode == "none":
            return None
        if offload_mode == "cpu":
            return CPUOffloadPolicy(pin_memory=bool(self.fsdp2_cpu_offload_pin_memory))
        raise ValueError(f"Unknown fsdp2_offload mode: {self.fsdp2_offload}")

    def _build_mixed_precision_policy(self) -> Optional["MixedPrecisionPolicy"]:
        """Build mixed precision policy."""
        if self.precision == "fp32":
            return None
        dtype = convert_to_dtype(self.precision)
        return MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=torch.float32, cast_forward_inputs=True)

    def setup_distributed(self, timeout=timedelta(minutes=60)) -> None:
        """Initialize distributed environment and device mesh."""
        if self.full_determinism:
            enable_full_determinism(self.seed)
        else:
            set_seed(self.seed)

        if self.args.local_rank != -1:
            os.environ["LOCAL_RANK"] = str(self.args.local_rank)
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        if local_rank != -1:
            torch.cuda.set_device(local_rank)

        dist.init_process_group(timeout=timeout)
        self.world_size = dist.get_world_size()
        self.ring_attn_rank = 0
        set_ring_attn_group(None)

        duplicate_factor = max(1, self.ring_attn_size) * max(1, self.ds_tensor_parallel_size)
        if self.world_size % duplicate_factor != 0:
            raise ValueError(
                f"world_size({self.world_size}) must be divisible by ring_attn_size({self.ring_attn_size}) * "
                f"ds_tensor_parallel_size({self.ds_tensor_parallel_size})."
            )
        self.dp_size = self.world_size // duplicate_factor

        self._create_device_mesh()

        if self.ring_attn_size > 1:
            self._setup_ring_attention()

        self._setup_gradient_accumulation()

        self.print(
            f"[fsdp2] world_size={self.world_size}, dp_size={self.dp_size}, "
            f"ring_attn_size={self.ring_attn_size}, ds_tensor_parallel_size={self.ds_tensor_parallel_size}, "
            f"accumulated_gradient={self.accumulated_gradient}, fsdp2_offload={self.fsdp2_offload}"
        )

    def _create_device_mesh(self) -> None:
        """Create device mesh with layout (dp, cp, tp).

        Dimension order (outer to inner) follows communication frequency:
        - dp: lowest frequency (gradient sync at step boundaries)
        - cp: medium frequency (ring attention per layer)
        - tp: highest frequency (intra-layer all-reduce)

        This ensures CP and TP groups have contiguous ranks for efficient communication.

        Supports:
        - Pure DP: 1D mesh ("dp",)
        - DP + CP: 2D mesh ("dp", "cp")
        - DP + TP: 2D mesh ("dp", "tp")
        - DP + CP + TP: 3D mesh ("dp", "cp", "tp")
        """
        cp_size = int(self.ring_attn_size) if self.ring_attn_size > 1 else 1
        tp_size = int(self.ds_tensor_parallel_size) if self.ds_tensor_parallel_size > 1 else 1

        if self.world_size != self.dp_size * cp_size * tp_size:
            raise RuntimeError(
                f"[fsdp2] invalid world_size decomposition: world_size={self.world_size}, "
                f"dp_size={self.dp_size}, cp_size={cp_size}, tp_size={tp_size}"
            )

        # Case 1: Pure DP (no CP, no TP)
        if cp_size == 1 and tp_size == 1:
            self.fsdp_device_mesh = init_device_mesh(
                "cuda",
                (self.dp_size,),
                mesh_dim_names=("dp",),
            )
            return

        # Case 2: DP + TP only (no CP)
        if cp_size == 1 and tp_size > 1:
            self.fsdp_device_mesh = init_device_mesh(
                "cuda",
                (self.dp_size, tp_size),
                mesh_dim_names=("dp", "tp"),
            )
            return

        # Case 3: DP + CP only (no TP)
        if tp_size == 1:
            self.fsdp_device_mesh = init_device_mesh(
                "cuda",
                (self.dp_size, cp_size),
                mesh_dim_names=("dp", "cp"),
            )
            return

        # Case 4: DP + CP + TP (full 3D mesh)
        self.fsdp_device_mesh = init_device_mesh(
            "cuda",
            (self.dp_size, cp_size, tp_size),
            mesh_dim_names=("dp", "cp", "tp"),
        )

    def _setup_ring_attention(self) -> None:
        """Setup ring attention process group."""
        group = self.fsdp_device_mesh["cp"].get_group()
        self.ring_attn_rank = dist.get_rank(group=group)
        set_ring_attn_group(group)

        try:
            from ring_flash_attn import substitute_hf_flash_attn

            ring_head_stride = getattr(self.args, "ring_head_stride", 1)
            substitute_hf_flash_attn(self.ring_attn_group, ring_head_stride)
        except Exception as exc:
            raise RuntimeError(
                "ring_attn_size > 1 requires ring_flash_attn. "
                "Please install ring_flash_attn or set --ring_attn_size 1."
            ) from exc

    def _setup_gradient_accumulation(self) -> None:
        """Compute gradient accumulation steps."""
        denom = max(1, self.micro_train_batch_size) * max(1, self.world_size)
        numerator = self.train_batch_size * max(1, self.ring_attn_size) * max(1, self.ds_tensor_parallel_size)
        self.accumulated_gradient = max(1, numerator // denom)
        if getattr(self.args, "use_dynamic_batch", False):
            self.accumulated_gradient = 1

    def _get_fsdp_mesh(self):
        """Get the mesh for FSDP sharding.

        Returns only the dp submesh (1D). CP and TP handle their own gradient
        synchronization internally (ring_flash_attn for CP, transformers TP for TP).
        """
        if not hasattr(self, "fsdp_device_mesh") or self.fsdp_device_mesh is None:
            return None

        mesh_dim_names = getattr(self.fsdp_device_mesh, "mesh_dim_names", None)
        if not mesh_dim_names:
            raise RuntimeError("[fsdp2] DeviceMesh must have named dims.")
        if "dp" not in mesh_dim_names:
            raise RuntimeError(f"[fsdp2] DeviceMesh is missing required 'dp' dim (got {mesh_dim_names}).")

        return self.fsdp_device_mesh["dp"]

    def _get_dp_group(self):
        """Get the process group for data-parallel communication (metrics, loss reduction)."""
        if not hasattr(self, "fsdp_device_mesh") or self.fsdp_device_mesh is None:
            return None

        mesh_dim_names = getattr(self.fsdp_device_mesh, "mesh_dim_names", None)
        if not mesh_dim_names:
            raise RuntimeError("[fsdp2] DeviceMesh must have named dims.")
        if "dp" not in mesh_dim_names:
            raise RuntimeError(f"[fsdp2] DeviceMesh is missing required 'dp' dim (got {mesh_dim_names}).")

        return self.fsdp_device_mesh["dp"].get_group()

    @property
    def ring_attn_group(self):
        """Get ring attention process group."""
        return get_ring_attn_group()

    def prepare(self, *models_or_model_optim_pairs: ModelOrModelOptimPair, is_rlhf=False):
        """Prepare models for FSDP2 training."""
        ret: List[ModelOrModelOptimPair] = []
        self.is_rlhf = is_rlhf

        for arg in models_or_model_optim_pairs:
            if isinstance(arg, tuple):
                model, optim_, scheduler = arg
                if model is None:
                    ret.append((None, None, None))
                    continue
                is_actor = isinstance(model, Actor)
                module = model.model if is_actor else model
                module = self._wrap_train_model(module)
                if is_actor:
                    model.model = module
                    ret.append((model, optim_, scheduler))
                else:
                    ret.append((module, optim_, scheduler))
            else:
                model = arg
                if model is None:
                    ret.append(model)
                    continue
                is_actor = isinstance(model, Actor)
                module = model.model if is_actor else model
                module = self._wrap_train_model(module)
                if is_actor:
                    model.model = module
                    ret.append(model)
                else:
                    ret.append(module)

        return ret[0] if len(ret) == 1 else ret

    def _wrap_train_model(self, model: nn.Module) -> nn.Module:
        """Wrap model with FSDP2 fully_shard."""
        try:
            fsdp_mesh = self._get_fsdp_mesh()
            if fsdp_mesh is None:
                raise RuntimeError(
                    "[fsdp2] Device mesh is not initialized. Make sure `strategy.setup_distributed()` "
                    "is called before `strategy.prepare()`."
                )
            self._maybe_fully_shard_children(model)
            if not isinstance(model, FSDPModule):
                model = fully_shard(
                    model,
                    mesh=fsdp_mesh,
                    reshard_after_forward=self.fsdp2_reshard_after_forward,
                    offload_policy=self._offload_policy,
                    mp_policy=self._mp_policy,
                )
            return model
        except Exception as exc:
            raise RuntimeError(
                "fully_shard() failed while constructing the FSDP2 model. "
                "Please double-check that the model supports composable FSDP."
            ) from exc

    def _maybe_fully_shard_children(self, module: nn.Module) -> None:
        """Apply FSDP2 fully_shard to transformer layers."""
        layer_cls_to_wrap = getattr(module, "_no_split_modules", None)
        fsdp_mesh = self._get_fsdp_mesh()
        if fsdp_mesh is None:
            raise RuntimeError(
                "[fsdp2] Device mesh is not initialized. Make sure `strategy.setup_distributed()` "
                "is called before `strategy.prepare()`."
            )

        if not layer_cls_to_wrap:
            for child in module.children():
                if isinstance(child, FSDPModule):
                    continue
                if any(p.requires_grad for p in child.parameters(recurse=True)):
                    fully_shard(
                        child,
                        mesh=fsdp_mesh,
                        reshard_after_forward=self.fsdp2_reshard_after_forward,
                        offload_policy=self._offload_policy,
                        mp_policy=self._mp_policy,
                    )
            return

        modules_to_shard = []
        for name, child in module.named_modules():
            if isinstance(child, FSDPModule):
                continue
            if child.__class__.__name__ in layer_cls_to_wrap:
                modules_to_shard.append(child)
            elif (
                isinstance(child, nn.Embedding) and hasattr(module, "config") and not module.config.tie_word_embeddings
            ):
                modules_to_shard.append(child)

        for child in modules_to_shard:
            fully_shard(
                child,
                mesh=fsdp_mesh,
                reshard_after_forward=self.fsdp2_reshard_after_forward,
                offload_policy=self._offload_policy,
                mp_policy=self._mp_policy,
            )

    def _unwrap_model(self, model) -> nn.Module:
        """Unwrap Actor wrapper, return the inner model."""
        if isinstance(model, Actor):
            return self._unwrap_model(model.model)
        return model

    def create_optimizer(self, model, **kwargs) -> Optimizer:
        """Create optimizer."""
        if isinstance(model, Actor):
            model = model.model
        if "foreach" not in kwargs:
            tp_size = int(getattr(self, "ds_tensor_parallel_size", 1) or 1)
            dp_size = int(getattr(self, "dp_size", 1) or 1)
            if tp_size > 1 and dp_size <= 1:
                kwargs["foreach"] = False
        return optim.AdamW(model.parameters(), **kwargs)

    def backward(self, loss: torch.Tensor, model: nn.Module, optimizer: optim.Optimizer, **kwargs) -> None:
        """Backward pass."""
        accumulation_steps = max(1, int(getattr(self, "accumulated_gradient", 1)))
        if accumulation_steps > 1:
            loss = loss / accumulation_steps
        loss.backward()

    def optimizer_step(
        self,
        optimizer: optim.Optimizer,
        model: nn.Module,
        scheduler,
        name="model",
        **kwargs,
    ) -> None:
        """Optimizer step."""
        micro_step_key = f"optim_micro_step_{name}"
        self.time_steps[micro_step_key] += 1
        accumulation_steps = max(1, int(getattr(self, "accumulated_gradient", 1)))
        if self.time_steps[micro_step_key] % accumulation_steps != 0:
            return

        unwrapped = self._unwrap_model(model)
        if self.max_norm and self.max_norm > 0:
            self._clip_grad_norm_dtensor_safe(unwrapped, float(self.max_norm))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()

    def _clip_grad_norm_dtensor_safe(
        self,
        model: nn.Module,
        max_norm: float,
        norm_type: float = 2.0,
        error_if_nonfinite: bool = False,
        foreach: Optional[bool] = None,
    ) -> float:
        """DTensor-safe global grad clipping.

        Uses torch.nn.utils.get_total_norm which automatically handles DTensor
        gradient norm computation across DP/FSDP/TP dimensions.
        Reference: TorchTitan's clip_grad_norm_ implementation.
        """
        if max_norm <= 0:
            return 0.0

        parameters = [p for p in model.parameters() if p.grad is not None]
        if not parameters:
            return 0.0

        # Non-distributed case: use standard clip_grad_norm_
        if not dist.is_initialized():
            total_norm = torch.nn.utils.clip_grad_norm_(
                parameters,
                max_norm=float(max_norm),
                norm_type=float(norm_type),
                error_if_nonfinite=bool(error_if_nonfinite),
                foreach=foreach,
            )
            return float(total_norm)

        # Distributed case: use get_total_norm + clip_grads_with_norm_
        grads = [p.grad for p in parameters]

        try:
            # get_total_norm handles DTensor automatically (PyTorch 2.4+)
            total_norm = torch.nn.utils.get_total_norm(
                grads,
                norm_type=float(norm_type),
                error_if_nonfinite=bool(error_if_nonfinite),
                foreach=foreach if foreach is not None else False,
            )

            # If total_norm is a DTensor, reduce it to get the global norm
            try:
                from torch.distributed.tensor import DTensor

                if isinstance(total_norm, DTensor):
                    total_norm = total_norm.full_tensor()
            except ImportError:
                pass
        except Exception:
            # Fallback for mixed DeviceMesh (e.g., TP parts vs non-TP parts)
            # PyTorch get_total_norm fails when stacking tensors with different meshes
            try:
                from torch.distributed.tensor import DTensor
            except Exception:
                DTensor = None

            mesh_to_grads: dict[tuple, list] = {}
            local_grads: list = []
            for grad in grads:
                if DTensor is not None and isinstance(grad, DTensor):
                    placements = getattr(grad, "placements", None)
                    placements_key = None
                    if placements is not None:
                        placements_key = tuple((p.__class__.__name__, getattr(p, "dim", None)) for p in placements)
                    key = (id(grad.device_mesh), placements_key)
                    mesh_to_grads.setdefault(key, []).append(grad)
                else:
                    local_grads.append(grad)

            norm_type_f = float(norm_type)
            if math.isinf(norm_type_f):
                total_norm = torch.zeros((), device=device)
            else:
                total_norm_pow = torch.zeros((), device=device)

            for group_grads in mesh_to_grads.values():
                group_norm = torch.nn.utils.get_total_norm(
                    group_grads,
                    norm_type=norm_type_f,
                    error_if_nonfinite=bool(error_if_nonfinite),
                    foreach=bool(foreach),
                )
                if DTensor is not None and isinstance(group_norm, DTensor):
                    group_norm = group_norm.full_tensor()
                group_norm = group_norm.to(device, non_blocking=True)

                if math.isinf(norm_type_f):
                    total_norm = torch.maximum(total_norm, group_norm)
                else:
                    total_norm_pow += group_norm.pow(norm_type_f)

            if local_grads:
                local_norm = torch.nn.utils.get_total_norm(
                    local_grads,
                    norm_type=norm_type_f,
                    error_if_nonfinite=bool(error_if_nonfinite),
                    foreach=bool(foreach),
                )
                if DTensor is not None and isinstance(local_norm, DTensor):
                    # Should not happen for local grads, but safe check
                    local_norm = local_norm.full_tensor()
                local_norm = local_norm.to(device, non_blocking=True)

                if math.isinf(norm_type_f):
                    total_norm = torch.maximum(total_norm, local_norm)
                else:
                    total_norm_pow += local_norm.pow(norm_type_f)

            if not math.isinf(norm_type_f):
                total_norm = total_norm_pow.pow(1.0 / norm_type_f)

        # Clip gradients using the computed total norm
        total_norm = total_norm.to(device, non_blocking=True)
        torch.nn.utils.clip_grads_with_norm_(
            parameters,
            max_norm=float(max_norm),
            total_norm=total_norm,
            foreach=foreach if foreach is not None else False,
        )

        return float(total_norm.item() if hasattr(total_norm, "item") else total_norm)


    @staticmethod
    def _move_optimizer_state(optimizer: optim.Optimizer, device: torch.device) -> None:
        """Move optimizer state to the specified device."""
        if not optimizer.state:
            return
        for param_group in optimizer.param_groups:
            for param in param_group["params"]:
                state = optimizer.state.get(param)
                if state is None:
                    continue
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device, non_blocking=True)
                    elif isinstance(v, (list, tuple)):
                        state[k] = type(v)(t.to(device, non_blocking=True) if torch.is_tensor(t) else t for t in v)
        if device.type == "cuda":
            torch.cuda.synchronize()

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
        """Setup dataloader."""
        if sampler is None and dist.is_initialized():
            dp_group = self._get_dp_group()
            num_replicas = dist.get_world_size(group=dp_group) if dp_group is not None else dist.get_world_size()
            rank = dist.get_rank(group=dp_group) if dp_group is not None else dist.get_rank()

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

    def offload_states(self, model, optimizer=None):
        """Offload training states to CPU."""
        unwrapped = self._unwrap_model(model)
        for module in unwrapped.modules():
            if isinstance(module, FSDPModule):
                module.reshard()
        if optimizer is not None:
            self._move_optimizer_state(optimizer, torch.device("cpu"))
        torch.cuda.empty_cache()
        if dist.is_initialized():
            dist.barrier()

    def reload_states(self, model, optimizer=None):
        """Reload training states to GPU."""
        device = torch.device("cuda", torch.cuda.current_device())
        self._unwrap_model(model).to(device)
        if optimizer is not None:
            self._move_optimizer_state(optimizer, device)
        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

    def moving_average(self, model, model_ema, beta=0.992, device="cpu"):
        """Update EMA model."""
        self.time_steps["ema"] += 1
        if self.time_steps["ema"] % max(1, self.accumulated_gradient) == 0:
            wrapped_model = self._unwrap_model(model)

            if not isinstance(wrapped_model, FSDPModule):
                raise RuntimeError(
                    "moving_average() must be called after strategy.prepare(). "
                    "The model is not an FSDPModule. Please call prepare() first."
                )

            full_state_dict = get_model_state_dict(
                model=wrapped_model,
                options=StateDictOptions(full_state_dict=True, cpu_offload=(device == "cpu")),
            )
            self._moving_average_from_state_dict(full_state_dict, model_ema, beta, device)

    def _moving_average_from_state_dict(self, state_dict, model_ema, beta, device):
        """Update EMA model using full state dict."""
        with torch.no_grad():
            for name, param_ema in model_ema.named_parameters():
                if not param_ema.requires_grad:
                    continue
                if name in state_dict:
                    param_data = state_dict[name]
                    param_ema.data.mul_(beta)
                    param_ema.data.add_(param_data.to(device), alpha=1 - beta)

    def load_model(
        self,
        model: nn.Module,
        path: str,
        map_location="cpu",
        strict: bool = False,
        key_replace_fn=None,
    ) -> None:
        """Load model weights."""
        state_dict = torch.load(path, map_location=map_location)
        if key_replace_fn:
            state_dict = key_replace_fn(state_dict)

        wrapped_model = self._unwrap_model(model)

        if not isinstance(wrapped_model, FSDPModule):
            raise RuntimeError(
                "load_model() must be called after strategy.prepare(). "
                "The model is not an FSDPModule. Please call prepare() first."
            )

        set_model_state_dict(
            model=wrapped_model,
            model_state_dict=state_dict,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True, strict=strict),
        )

    def save_model(self, model: nn.Module, tokenizer, output_dir, **kwargs) -> None:
        """Save model to HuggingFace format."""
        if self.is_rank_0():
            os.makedirs(output_dir, exist_ok=True)

        fsdp_model = self._unwrap_model(model)
        is_peft_model = hasattr(fsdp_model, "peft_config")

        if self.is_rank_0():
            config = getattr(fsdp_model, "config", None)
            if config is not None and hasattr(config, "auto_map") and isinstance(config.auto_map, dict):
                if None in config.auto_map:
                    config.auto_map = {k: v for k, v in config.auto_map.items() if k is not None}

        model_state = get_model_state_dict(
            model=fsdp_model,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True, ignore_frozen_params=is_peft_model),
        )

        if self.is_rank_0():
            fsdp_model.save_pretrained(output_dir, state_dict=model_state, **kwargs)
            self._save_model_configs(fsdp_model, output_dir, tokenizer)

        if dist.is_initialized():
            dist.barrier()

        del model_state
        import gc

        gc.collect()

    def _save_model_configs(self, fsdp_model, output_dir, tokenizer):
        """Save model configs and related files."""
        config = getattr(fsdp_model, "config", None)
        if config is not None:
            try:
                config.save_pretrained(output_dir)
            except Exception:
                config.to_json_file(os.path.join(output_dir, "config.json"))

            try:
                if getattr(config, "auto_map", None):
                    from transformers.dynamic_module_utils import custom_object_save

                    custom_object_save(fsdp_model, output_dir, config=config)
            except Exception as exc:
                self.print(f"[fsdp2] warning: failed to save custom code: {exc}")

        generation_config = getattr(fsdp_model, "generation_config", None)
        if generation_config is not None:
            try:
                generation_config.save_pretrained(output_dir)
            except Exception:
                pass

        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)

        try:
            import json

            metadata = {
                "backend": "fsdp2",
                "world_size": int(getattr(self, "world_size", 1) or 1),
                "dp_size": int(getattr(self, "dp_size", getattr(self, "world_size", 1)) or 1),
                "ring_attn_size": int(getattr(self, "ring_attn_size", 1) or 1),
                "ds_tensor_parallel_size": int(getattr(self, "ds_tensor_parallel_size", 1) or 1),
                "fsdp2_offload": str(getattr(self, "fsdp2_offload", "none") or "none"),
                "fsdp2_reshard_after_forward": bool(getattr(self, "fsdp2_reshard_after_forward", True)),
                "precision": self.precision,
            }
            with open(os.path.join(output_dir, "fsdp2_runtime.json"), "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, sort_keys=True)
        except Exception:
            pass

    def save_ckpt(self, *args, **kwargs):
        """Save distributed checkpoint."""
        return self._save_ckpt_impl(*args, **kwargs)

    def load_ckpt(self, *args, **kwargs):
        """Load distributed checkpoint."""
        return self._load_ckpt_impl(*args, **kwargs)

    def _save_ckpt_impl(
        self,
        model: nn.Module,
        save_dir: str,
        tag: Optional[str] = None,
        max_num: int = 3,
        max_mem: int = 1000,
        client_state: Optional[dict] = None,
        save_latest: bool = True,
        optimizer: Optional[Optimizer] = None,
        scheduler=None,
        **kwargs,
    ):
        """Save FSDP2 distributed checkpoint."""
        import warnings
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import get_state_dict
        from torch.distributed.checkpoint.stateful import Stateful

        if tag is None:
            raise ValueError("FSDP2 save_ckpt requires a non-empty tag (e.g., 'global_step123').")

        os.makedirs(save_dir, exist_ok=True)

        if self.is_rank_0():
            self._cleanup_old_checkpoints(save_dir, max_num, max_mem)

        if dist.is_initialized():
            dist.barrier()

        fsdp_model = self._unwrap_model(model)
        ckpt_path = os.path.join(save_dir, tag)

        class AppState(Stateful):
            def __init__(self, model_, optimizer_, scheduler_, client_state_):
                self.model = model_
                self.optimizer = optimizer_
                self.scheduler = scheduler_
                self.client_state = dict(client_state_ or {})

            def state_dict(self):
                optimizers = [self.optimizer] if self.optimizer is not None else []
                model_state, optim_state = get_state_dict(self.model, optimizers)
                state = {
                    "model": model_state,
                    "optimizers": optim_state,
                    "client_state": self.client_state,
                }
                if self.scheduler is not None:
                    state["scheduler"] = self.scheduler.state_dict()
                return state

            def load_state_dict(self, state_dict):
                raise RuntimeError("AppState.load_state_dict should not be called from save_ckpt().")

        state_dict = {"app": AppState(fsdp_model, optimizer, scheduler, client_state)}
        if self.is_rank_0():
            os.makedirs(ckpt_path, exist_ok=True)

        if dist.is_initialized():
            dist.barrier()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="torch.distributed")
            warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributed.*")
            dcp.save(state_dict=state_dict, checkpoint_id=ckpt_path)

        if dist.is_initialized():
            dist.barrier()

        if save_latest and self.is_rank_0():
            latest_path = os.path.join(save_dir, "latest")
            with open(latest_path, "w", encoding="utf-8") as f:
                f.write(str(tag))

    def _cleanup_old_checkpoints(self, save_dir: str, max_num: int, max_mem: int):
        """Cleanup old checkpoints."""
        import shutil

        max_size_bytes = max_mem * 1024**3

        def _list_ckpt_dirs():
            entries = []
            for name in os.listdir(save_dir):
                path = os.path.join(save_dir, name)
                if os.path.isdir(path):
                    entries.append((path, os.path.getmtime(path)))
            entries.sort(key=lambda x: x[1])
            return entries

        def _dir_size_bytes(path: str) -> int:
            total = 0
            for dirpath, _dirnames, filenames in os.walk(path):
                for filename in filenames:
                    fp = os.path.join(dirpath, filename)
                    try:
                        total += os.path.getsize(fp)
                    except OSError:
                        pass
            return total

        while True:
            subdirs = _list_ckpt_dirs()
            total_size = sum(_dir_size_bytes(subdir) for subdir, _ in subdirs)
            if len(subdirs) >= max_num or total_size > max_size_bytes:
                oldest_dir = subdirs[0][0] if subdirs else None
                if oldest_dir and os.path.exists(oldest_dir):
                    shutil.rmtree(oldest_dir, ignore_errors=True)
                    self.print(f"Deleted oldest ckpt {oldest_dir}")
                else:
                    break
            else:
                break

    def _load_ckpt_impl(
        self,
        model: nn.Module,
        load_dir: str,
        tag: Optional[str] = None,
        load_module_strict: bool = True,
        load_optimizer_states: bool = True,
        load_lr_scheduler_states: bool = True,
        load_module_only: bool = False,
        optimizer: Optional[Optimizer] = None,
        scheduler=None,
        **kwargs,
    ):
        """Load FSDP2 distributed checkpoint."""
        import warnings
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import StateDictOptions, set_state_dict
        from torch.distributed.checkpoint.stateful import Stateful

        if not os.path.exists(load_dir):
            raise FileNotFoundError(f"[fsdp2] checkpoint directory not found: {load_dir}")

        resolved_tag = tag
        if resolved_tag is None:
            latest_path = os.path.join(load_dir, "latest")
            if os.path.isfile(latest_path):
                with open(latest_path, "r", encoding="utf-8") as f:
                    resolved_tag = f.read().strip()
            else:
                subdirs = [
                    os.path.join(load_dir, d) for d in os.listdir(load_dir) if os.path.isdir(os.path.join(load_dir, d))
                ]
                if subdirs:
                    subdirs.sort(key=lambda p: os.path.getmtime(p))
                    resolved_tag = os.path.basename(subdirs[-1])

        if not resolved_tag:
            raise FileNotFoundError(f"[fsdp2] no checkpoint tag found under {load_dir}")

        ckpt_path = os.path.join(load_dir, resolved_tag)
        if not os.path.isdir(ckpt_path):
            raise FileNotFoundError(f"[fsdp2] checkpoint path not found: {ckpt_path}")

        fsdp_model = self._unwrap_model(model)

        if load_module_only:
            load_optimizer_states = False
            load_lr_scheduler_states = False
        if optimizer is None:
            load_optimizer_states = False
        if scheduler is None:
            load_lr_scheduler_states = False

        class AppState(Stateful):
            def __init__(self, model_, optimizer_, scheduler_):
                self.model = model_
                self.optimizer = optimizer_
                self.scheduler = scheduler_
                self.client_state: dict = {}

            def state_dict(self):
                raise RuntimeError("AppState.state_dict should not be called from load_ckpt().")

            def load_state_dict(self, state_dict):
                optimizers = [self.optimizer] if (load_optimizer_states and self.optimizer is not None) else []
                if optimizers:
                    optim_state = state_dict.get("optimizers")
                    if optim_state is None:
                        raise RuntimeError(
                            "[fsdp2] checkpoint is missing optimizer states, but load_optimizer_states=True."
                        )
                else:
                    optim_state = {}
                set_state_dict(
                    self.model,
                    optimizers,
                    model_state_dict=state_dict.get("model"),
                    optim_state_dict=optim_state,
                    options=StateDictOptions(strict=bool(load_module_strict)),
                )
                if load_lr_scheduler_states and self.scheduler is not None and "scheduler" in state_dict:
                    self.scheduler.load_state_dict(state_dict["scheduler"])
                self.client_state = state_dict.get("client_state", {}) or {}

        app_state = AppState(fsdp_model, optimizer, scheduler)
        state_dict = {"app": app_state}

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="torch.distributed")
            warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributed.*")
            dcp.load(state_dict=state_dict, checkpoint_id=ckpt_path)

        if dist.is_initialized():
            dist.barrier()

        return ckpt_path, app_state.client_state

    def all_reduce(self, data, op="mean"):
        """All-reduce operation."""
        assert op in ("mean", "max", "sum")
        if isinstance(data, dict):
            return {k: self.all_reduce(v, op) for k, v in data.items()}

        is_tensor = True
        if not isinstance(data, torch.Tensor):
            data = torch.tensor([data])
            is_tensor = False
        is_cpu_tensor = data.device.type == "cpu"

        dp_group = self._get_dp_group()
        dp_world_size = dist.get_world_size(group=dp_group) if dist.is_initialized() else 1
        if is_cpu_tensor:
            data = data.to(torch.cuda.current_device())
        if op == "mean":
            data = data / max(1, dp_world_size)
        dist.all_reduce(
            data,
            op=dist.ReduceOp.MAX if op == "max" else dist.ReduceOp.SUM,
            group=dp_group,
        )
        if is_cpu_tensor:
            data = data.cpu()
        return data.item() if not is_tensor else data

    def all_gather(self, data):
        """All-gather operation."""
        if isinstance(data, dict):
            return {k: self.all_gather(v) for k, v in data.items()}

        is_tensor = True
        if not isinstance(data, torch.Tensor):
            data = torch.tensor([data])
            is_tensor = False
        is_cpu_tensor = data.device.type == "cpu"

        dp_group = self._get_dp_group()
        dp_world_size = dist.get_world_size(group=dp_group) if dist.is_initialized() else 1
        ret = [torch.zeros_like(data).to(torch.cuda.current_device()) for _ in range(dp_world_size)]
        dist.all_gather(ret, data.to(torch.cuda.current_device()), group=dp_group)
        result = torch.cat(ret).cpu() if is_cpu_tensor else torch.cat(ret)
        if not is_tensor:
            return result.tolist()
        return result

    def print(self, *msg):
        """Print only on rank 0."""
        if self.is_rank_0():
            print(*msg)

    def is_rank_0(self) -> bool:
        """Check if current rank is 0."""
        if not dist.is_initialized():
            return True
        return dist.get_rank() == 0

    def get_rank(self) -> int:
        """Get current rank."""
        if not dist.is_initialized():
            return 0
        return dist.get_rank()

    def get_ds_train_config(self, is_actor):
        return None

    def get_ds_eval_config(self, offload=False):
        return None
