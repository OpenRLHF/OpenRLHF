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

CP loss scaling note:
- Ring Attention's all_gather autograd uses reduce_scatter(SUM) in backward,
  introducing a cp_size factor into local gradients.
- FSDP2 averages gradients across the flattened dp_cp mesh, dividing by
  (dp_size * cp_size); the cp_size factor cancels out, yielding an effective
  "dp average, cp sum" without explicitly scaling the loss.
- MoE aux_loss needs no special handling: FSDP2's AVG across dp_cp correctly
  averages the per-rank load balance metric.

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
from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffloadPolicy, FSDPModule, MixedPrecisionPolicy, fully_shard
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import enable_full_determinism, set_seed

from openrlhf.models import Actor
from openrlhf.models.ring_attn_utils import get_ring_attn_group, set_ring_attn_group, set_ring_attn_pad_multiple
from openrlhf.utils import convert_to_torch_dtype
from openrlhf.utils.distributed_sampler import DistributedSampler

from .checkpoint import (
    _cleanup_old_checkpoints,
    _load_hf_checkpoint,
    _save_hf_checkpoint,
    _load_dcp_checkpoint,
    _save_dcp_checkpoint,
)
from .tp.tp_parallel import apply_tensor_parallel
from .utils import (
    clip_grad_norm_dtensor,
    get_checkpoint_metadata,
    move_optimizer_state,
    moving_average_fsdp2,
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
        self.fsdp2_cp_size = args.fsdp2_cp_size
        self.fsdp2_tp_size = args.fsdp2_tp_size
        self.fsdp2_tp_loss_parallel = args.fsdp2_tp_loss_parallel

        # fsdp2_tp_loss_parallel requires TP > 1 (lm_head outputs vocab-sharded DTensor logits)
        if self.fsdp2_tp_loss_parallel and self.fsdp2_tp_size <= 1:
            raise ValueError("--fsdp2_tp_loss_parallel requires --fsdp2_tp_size > 1.")

        # FSDP config
        self.param_dtype = args.param_dtype
        self.fsdp2_cpu_offload = args.fsdp2_cpu_offload
        self.fsdp2_reshard_after_forward = args.fsdp2_reshard_after_forward
        self.sequence_parallel = args.fsdp2_tp_sequence_parallel

        # CPUOffloadPolicy and manual offload (fsdp2_enable_sleep) are mutually exclusive:
        # FSDP2 manages CPU offload automatically when fsdp2_cpu_offload is enabled.
        if self.fsdp2_cpu_offload and getattr(args, "fsdp2_enable_sleep", False):
            args.fsdp2_enable_sleep = False
            print("[FSDP2] Warning: fsdp2_enable_sleep disabled because fsdp2_cpu_offload is enabled")

        # State
        self.time_steps = defaultdict(int)
        self.mesh = None
        self._gloo_group = None

        # Derive checkpoint sub-paths from ckpt_save_path
        ckpt_save_path = getattr(args, "ckpt_save_path", None)
        if ckpt_save_path is not None:
            self.last_hf_ckpt_path = os.path.join(ckpt_save_path, "last_hf_ckpt")
            self.hf_ckpt_path = os.path.join(ckpt_save_path, "hf_ckpt")
            self.dcp_ckpt_path = os.path.join(ckpt_save_path, "dcp_ckpt")

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

        # Explicit backend + device_id reduces NCCL barrier warnings and avoids
        # hangs when global-rank-to-GPU mapping is heterogeneous.
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        device_id = local_rank if (backend == "nccl" and local_rank >= 0) else None
        dist.init_process_group(backend=backend, timeout=timeout, device_id=device_id)
        self.world_size = dist.get_world_size()

        # Gloo group for CPU-safe barriers: NCCL requires GPU tensors, which
        # fail when model is offloaded to CPU.
        self._gloo_group = dist.new_group(backend="gloo")

        # Validate and compute parallelism sizes
        cp_tp_factor = self.fsdp2_cp_size * self.fsdp2_tp_size
        assert (
            self.world_size % cp_tp_factor == 0
        ), f"world_size({self.world_size}) not divisible by cp*tp({cp_tp_factor})"
        self.fsdp2_dp_size = self.world_size // cp_tp_factor

        # Sequence Parallel (SP) is only meaningful with TP>1.
        if self.sequence_parallel:
            if self.fsdp2_tp_size <= 1:
                raise ValueError("Invalid config: --fsdp2_tp_sequence_parallel requires --fsdp2_tp_size > 1.")
            if not getattr(self.args, "packing_samples", False):
                raise ValueError(
                    "--fsdp2_tp_sequence_parallel requires --packing_samples to be enabled, "
                    "because HF's causal mask creation uses inputs_embeds.shape[1] "
                    "which would be seq/tp_size after SP sharding, causing mask length mismatch."
                )

        # Always create 3D mesh even if some dims are size=1, so all parameters
        # share the same mesh structure (avoids mismatch in clip_grad_norm etc.)
        self.mesh = init_device_mesh(
            "cuda",
            (self.fsdp2_dp_size, self.fsdp2_cp_size, self.fsdp2_tp_size),
            mesh_dim_names=("dp", "cp", "tp"),
        )

        # Merge DP+CP for FSDP reduce-scatter (see module docstring for rationale)
        self.fsdp_mesh = self.mesh["dp", "cp"]._flatten(mesh_dim_name="dp_cp")

        # Process groups:
        #   dp_cp_group — metrics reduction across all FSDP ranks
        #   dp_group    — data sampling (CP ranks share same data)
        #   cp_group    — Ring Attention KV communication
        self.dp_cp_group = self.fsdp_mesh.get_group()
        self.dp_group = self.mesh["dp"].get_group()
        self.cp_group = self.mesh["cp"].get_group()

        # Initialize ring attention defaults
        set_ring_attn_group(None)
        set_ring_attn_pad_multiple(1)

        if self.sequence_parallel and self.fsdp2_tp_size > 1:
            # Pad packed sequence length to be divisible by TP degree for SP
            set_ring_attn_pad_multiple(self.fsdp2_tp_size)

        if self.fsdp2_cp_size > 1:
            set_ring_attn_group(self.cp_group)
            try:
                from ring_flash_attn import substitute_hf_flash_attn
            except ModuleNotFoundError as e:  # pragma: no cover
                raise RuntimeError(
                    "ring_flash_attn is required when --fsdp2_cp_size > 1. "
                    "Install ring_flash_attn or set --fsdp2_cp_size 1."
                ) from e

            substitute_hf_flash_attn(self.cp_group, getattr(self.args, "ring_head_stride", 1))

        # Gradient accumulation: only DP contributes to effective batch size
        batch_per_step = self.micro_train_batch_size * self.fsdp2_dp_size
        if getattr(self.args, "use_dynamic_batch", False):
            self.accumulated_gradient = 1
        else:
            accum_steps, remainder = divmod(self.train_batch_size, batch_per_step)
            if accum_steps < 1 or remainder != 0:
                raise ValueError(
                    "Invalid batch config for FSDP2: require "
                    "`train_batch_size = micro_train_batch_size * fsdp2_dp_size * grad_accum_steps` "
                    f"(got train_batch_size={self.train_batch_size}, "
                    f"micro_train_batch_size={self.micro_train_batch_size}, fsdp2_dp_size={self.fsdp2_dp_size})."
                )
            self.accumulated_gradient = accum_steps

        self.print(
            f"[FSDP2] world={self.world_size} dp={self.fsdp2_dp_size} cp={self.fsdp2_cp_size} "
            f"tp={self.fsdp2_tp_size} grad_accum={self.accumulated_gradient} "
            f"fsdp_mesh_size={self.fsdp2_dp_size * self.fsdp2_cp_size}"
        )

        # TP-aware loss is handled by DTensor-aware helpers (no monkey patch needed).

    @property
    def ring_attn_group(self):
        return get_ring_attn_group()

    # -------------------------------------------------------------------------
    # Model Sharding
    # -------------------------------------------------------------------------

    def apply_parallelism(self, model, force_cpu_offload: bool = False):
        """Apply TP then FSDP sharding to a model, preserving Actor wrapper.

        Args:
            model: Model to shard.
            force_cpu_offload: If True, enable CPU offload policy for this model.
        """
        unwrapped = self._unwrap_model(model)
        is_actor = unwrapped is not model

        self.print(
            f"[FSDP2] Sharding model: dp={self.fsdp2_dp_size} cp={self.fsdp2_cp_size} tp={self.fsdp2_tp_size} "
            f"(fsdp_mesh_size={self.fsdp2_dp_size * self.fsdp2_cp_size})"
        )
        if self.fsdp2_tp_size > 1:
            self.print(f"[FSDP2] Applying TP (size={self.fsdp2_tp_size})")
            unwrapped = apply_tensor_parallel(
                unwrapped,
                self.mesh["tp"],
                sequence_parallel=self.sequence_parallel,
                validate=True,
                shard_logits=self.fsdp2_tp_loss_parallel,
            )
        else:
            self.print("[FSDP2] Skipping TP (fsdp2_tp_size=1)")

        unwrapped = self._apply_fsdp(unwrapped, force_cpu_offload=force_cpu_offload)

        if is_actor:
            model.model = unwrapped
            return model
        return unwrapped

    def _apply_fsdp(self, model, force_cpu_offload=False):
        """Apply FSDP2 sharding using merged DP+CP mesh (see module docstring).

        Args:
            model: Model to shard.
            force_cpu_offload: If True, force CPUOffloadPolicy regardless of self.fsdp2_cpu_offload.
        """
        mesh = self.fsdp_mesh
        mixed_precision = (
            None
            if self.param_dtype == "fp32"
            else MixedPrecisionPolicy(
                param_dtype=convert_to_torch_dtype(self.param_dtype),
                reduce_dtype=torch.float32,
                cast_forward_inputs=True,
            )
        )
        use_cpu_offload = force_cpu_offload or self.fsdp2_cpu_offload
        offload_policy = CPUOffloadPolicy(pin_memory=True) if use_cpu_offload else None

        # Collect modules that should be individually sharded (transformer layers + untied embeddings)
        no_split_modules = getattr(model, "_no_split_modules", [])
        shard_units = [
            m
            for m in model.modules()
            if m.__class__.__name__ in no_split_modules
            or (
                isinstance(m, nn.Embedding)
                and not getattr(getattr(model, "config", None), "tie_word_embeddings", True)
            )
        ]

        for i, layer in enumerate(shard_units):
            if not isinstance(layer, FSDPModule):
                fully_shard(
                    layer,
                    mesh=mesh,
                    mp_policy=mixed_precision,
                    offload_policy=offload_policy,
                    reshard_after_forward=self.fsdp2_reshard_after_forward and i < len(shard_units) - 1,
                )

        if not isinstance(model, FSDPModule):
            fully_shard(model, mesh=mesh, mp_policy=mixed_precision, offload_policy=offload_policy, reshard_after_forward=False)
        return model

    # -------------------------------------------------------------------------
    # Model Loading & Post-load Fixup
    # -------------------------------------------------------------------------

    def load_hf_checkpoint(
        self,
        model: nn.Module,
        model_name_or_path: str,
        *,
        force_cpu_offload: bool = False,
        init_value_head: bool = False,
        value_head_prefix: str = "score",
    ) -> bool:
        """Materialize (to_empty) and load pretrained weights into an already wrapped model.

        Returns:
          True on success.
        """
        unwrapped = self._unwrap_model(model)

        load_to_cpu = self.fsdp2_cpu_offload or force_cpu_offload
        device = "cpu" if load_to_cpu else torch.device("cuda", torch.cuda.current_device())

        _load_hf_checkpoint(
            unwrapped,
            model_name_or_path,
            device=device,
            process_group=self._gloo_group,
        )

        if load_to_cpu:
            self._move_buffers_to_cuda(unwrapped)
        if init_value_head:
            self._init_value_head_after_load(unwrapped, value_head_prefix=value_head_prefix)
        return True

    @torch.no_grad()
    def _move_buffers_to_cuda(self, model: nn.Module) -> None:
        """Keep buffers on CUDA for CPU-offloaded FSDP2 models."""
        device = torch.device("cuda", torch.cuda.current_device())
        for _name, buf in model.named_buffers():
            if getattr(buf, "is_meta", False) or buf.device == device:
                continue
            buf.data = buf.data.to(device)

    @torch.no_grad()
    def _init_value_head_after_load(self, model: nn.Module, value_head_prefix: str) -> None:
        """Initialize value head weights after meta-init materialization."""
        value_head = getattr(model, value_head_prefix, None)
        if value_head is None or not hasattr(value_head, "weight"):
            return

        weight = value_head.weight
        weight_key = f"{value_head_prefix}.weight"
        if dist.is_initialized() and dist.get_rank() != 0:
            state = {}
        else:
            shape = tuple(weight.shape)
            dtype = weight.dtype
            hidden_size = shape[1] if len(shape) >= 2 else max(1, shape[0])
            # Match existing init convention: std = 1 / (hidden_size + 1)
            std = 1.0 / float(hidden_size + 1)
            init_weight = torch.empty(shape, device="cpu", dtype=dtype)
            init_weight.normal_(mean=0.0, std=std)
            state = {weight_key: init_weight}

        set_model_state_dict(
            model,
            state,
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
                strict=False,
                broadcast_from_rank0=dist.is_initialized(),
            ),
        )

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    def create_optimizer(self, model, **kwargs):
        """Create AdamW optimizer."""
        if "foreach" not in kwargs and self.fsdp2_tp_size > 1 and self.fsdp2_dp_size <= 1:
            kwargs["foreach"] = False
        weight_decay = kwargs.pop("weight_decay", 0.0)
        param_groups = self._get_optimizer_grouped_parameters(self._unwrap_model(model), weight_decay)
        # fused=True only works on CUDA and is incompatible with CPU offload
        fused = torch.cuda.is_available() and not self.fsdp2_cpu_offload
        return optim.AdamW(param_groups, fused=fused, **kwargs)

    @staticmethod
    def _get_optimizer_grouped_parameters(
        model: nn.Module,
        weight_decay: float,
        no_decay_name_list=("bias", "layer_norm.weight", "layernorm.weight", "norm.weight", "ln_f.weight"),
    ):
        """Separate parameters into weight-decay and no-weight-decay groups."""
        decay = []
        no_decay = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name for nd in no_decay_name_list):
                no_decay.append(param)
            else:
                decay.append(param)
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    def backward(self, loss, model, optimizer, name="model", **kwargs):
        """Backward with gradient accumulation.

        No explicit CP loss scaling is needed here — see module docstring for
        why the all_gather reduce_scatter factor cancels FSDP's dp_cp averaging.
        """
        if self.accumulated_gradient > 1:
            loss = loss / self.accumulated_gradient

        unwrapped = self._unwrap_model(model)
        if isinstance(unwrapped, FSDPModule) and self.accumulated_gradient > 1:
            key = f"step_{name}"
            is_final = (self.time_steps.get(key, 0) + 1) % self.accumulated_gradient == 0
            unwrapped.set_requires_gradient_sync(is_final)

        loss.backward()

    def optimizer_step(self, optimizer, model, scheduler, name="model", **kwargs):
        """Optimizer step with gradient accumulation."""
        key = f"step_{name}"
        self.time_steps[key] += 1
        if self.time_steps[key] % self.accumulated_gradient != 0:
            return

        if self.max_norm > 0:
            # DTensor-compatible clipping (standard clip_grad_norm_ fails with mixed meshes)
            clip_grad_norm_dtensor(self._unwrap_model(model), max_norm=self.max_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if scheduler:
            scheduler.step()

    def moving_average(self, model, model_ema, beta=0.992, device="cpu"):
        """Update EMA model."""
        self.time_steps["ema"] += 1
        if self.time_steps["ema"] % max(1, self.accumulated_gradient) == 0:
            moving_average_fsdp2(model, model_ema, self._unwrap_model, beta)

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

        Data is sharded only across DP, not CP. CP ranks within the same DP
        group see identical samples but process different sequence chunks.
        """
        if sampler is None and dist.is_initialized():
            dp_group = self.dp_group if self.dp_group else None
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

    @torch.no_grad()
    def offload_model(self, model):
        """Offload model params to CPU (hybrid engine: free GPU for vLLM rollout)."""
        if self.fsdp2_cpu_offload:
            return  # CPUOffloadPolicy already manages offload

        unwrapped = self._unwrap_model(model)
        unwrapped.cpu()
        self.print("[FSDP2] Model offloaded to CPU")
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def reload_model(self, model):
        """Reload model params to GPU (hybrid engine: before forward pass)."""
        if self.fsdp2_cpu_offload:
            return  # CPUOffloadPolicy already manages offload

        unwrapped = self._unwrap_model(model)
        if next(unwrapped.parameters()).device.type == "cpu":
            unwrapped.to(torch.device("cuda", torch.cuda.current_device()))
            torch.cuda.synchronize()
            self.print("[FSDP2] Model reloaded to GPU")

    @torch.no_grad()
    def offload_optimizer_states(self, optimizer):
        """Offload optimizer states to CPU (model stays on GPU for vLLM weight sync)."""
        move_optimizer_state(optimizer, torch.device("cpu"))
        self.print("[FSDP2] Optimizer states offloaded to CPU")

        torch.cuda.empty_cache()
        if self._gloo_group is not None:
            dist.barrier(group=self._gloo_group)
        torch.cuda.synchronize()

    @torch.no_grad()
    def reload_optimizer_states(self, optimizer):
        """Reload optimizer states to GPU."""
        device = torch.device("cuda", torch.cuda.current_device())
        move_optimizer_state(optimizer, device)
        self.print("[FSDP2] Optimizer states reloaded to GPU")

        torch.cuda.synchronize()
        if self._gloo_group is not None:
            dist.barrier(group=self._gloo_group)

    # -------------------------------------------------------------------------
    # Checkpointing
    # -------------------------------------------------------------------------

    def save_hf_checkpoint(self, model, tokenizer, save_dir):
        """Save HF checkpoint to an exact target directory."""

        # Convert fp32 master weights to the training compute dtype (e.g. bf16) for deployment
        save_dtype = convert_to_torch_dtype(self.param_dtype)
        max_gb = getattr(self.args, "hf_max_shard_size_gb", 5)
        max_bytes = int(float(max_gb) * 1024**3) if max_gb else None

        _save_hf_checkpoint(
            self._unwrap_model(model),
            tokenizer,
            save_dir,
            self.is_rank_0(),
            save_dtype=save_dtype,
            process_group=self._gloo_group,
            max_shard_size_bytes=max_bytes,
            metadata=get_checkpoint_metadata(self),
        )

    def save_dcp_checkpoint(
        self,
        model,
        save_dir,
        client_state=None,
        optimizer=None,
        scheduler=None,
    ):
        """Save FSDP2 distributed checkpoint."""
        _save_dcp_checkpoint(
            model,
            save_dir,
            self._unwrap_model,
            self.is_rank_0(),
            optimizer=optimizer,
            scheduler=scheduler,
            client_state=client_state,
            process_group=self._gloo_group,
        )

    def load_dcp_checkpoint(
        self,
        model,
        load_dir,
        load_module_strict=True,
        load_optimizer_states=True,
        load_lr_scheduler_states=True,
        load_module_only=False,
        optimizer=None,
        scheduler=None,
    ):
        return _load_dcp_checkpoint(
            model,
            load_dir,
            self._unwrap_model,
            optimizer=optimizer,
            scheduler=scheduler,
            load_optimizer_states=load_optimizer_states,
            load_lr_scheduler_states=load_lr_scheduler_states,
            load_module_strict=load_module_strict,
            load_module_only=load_module_only,
            process_group=self._gloo_group,
        )

    def cleanup_old_checkpoints(self, tag: str):
        """Write top-level latest marker and clean old step directories."""
        _cleanup_old_checkpoints(
            self.dcp_ckpt_path,
            self.args.max_checkpoints_to_keep,
            tag=tag,
            is_rank_0=self.is_rank_0(),
        )

    # -------------------------------------------------------------------------
    # Communication
    # -------------------------------------------------------------------------

    def all_reduce(self, data, op="mean", with_context_parallel=True):
        """All-reduce across DP group (optionally including CP).

        Args:
            data: Data to reduce (tensor, scalar, or dict).
            op: Reduction operation ("mean", "max", "sum").
            with_context_parallel: If True, reduce across DP+CP (for metrics).
                                   If False, reduce across DP only.
        """
        assert op in ("mean", "max", "sum")

        if isinstance(data, dict):
            return {k: self.all_reduce(v, op, with_context_parallel) for k, v in data.items()}

        if not dist.is_initialized():
            return data

        process_group = self.dp_cp_group if with_context_parallel and self.fsdp2_cp_size > 1 else self.dp_group
        group_size = dist.get_world_size(group=process_group)

        was_tensor = isinstance(data, torch.Tensor)
        tensor = data if was_tensor else torch.tensor(data, device="cuda")

        from_cpu = tensor.device.type == "cpu"
        if from_cpu:
            tensor = tensor.cuda()

        reduce_op = {"mean": dist.ReduceOp.SUM, "max": dist.ReduceOp.MAX, "sum": dist.ReduceOp.SUM}[op]
        dist.all_reduce(tensor, op=reduce_op, group=process_group)
        if op == "mean":
            tensor = tensor / group_size

        if from_cpu:
            tensor = tensor.cpu()

        return tensor if was_tensor else tensor.item()

    def all_gather(self, data):
        """All-gather across DP group (not CP/TP, as they share same data)."""
        if isinstance(data, dict):
            return {k: self.all_gather(v) for k, v in data.items()}

        if not dist.is_initialized():
            return data

        process_group = self.dp_group
        group_size = dist.get_world_size(group=process_group)

        was_tensor = isinstance(data, torch.Tensor)
        tensor = data if was_tensor else torch.tensor(data, device="cuda")

        if tensor.dim() == 0:
            tensor = tensor.view(1)

        from_cpu = tensor.device.type == "cpu"
        gpu_tensor = tensor.cuda() if from_cpu else tensor

        shards = [torch.zeros_like(gpu_tensor) for _ in range(group_size)]
        dist.all_gather(shards, gpu_tensor, group=process_group)
        result = torch.cat(shards)

        if from_cpu:
            result = result.cpu()

        return result if was_tensor else result.tolist()

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def print(self, *msg):
        """Rank-0 logging without prefix (public API, 30+ external callers)."""
        if self.is_rank_0():
            print(*msg)

    def is_rank_0(self):
        return not dist.is_initialized() or dist.get_rank() == 0

    def get_rank(self):
        return dist.get_rank() if dist.is_initialized() else 0

    def get_world_size(self):
        return dist.get_world_size() if dist.is_initialized() else 1

    def _unwrap_model(self, model):
        """Unwrap Actor wrapper to get the underlying module."""
        if isinstance(model, Actor):
            return self._unwrap_model(model.model)
        return model
