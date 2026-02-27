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
    _load_hf_checkpoint,
    _save_hf_checkpoint,
    _load_dcp_checkpoint,
    _save_dcp_checkpoint,
    _cleanup_old_checkpoints,
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
        self.tp_loss_parallel = args.tp_loss_parallel

        # tp_loss_parallel means lm_head outputs vocab-sharded DTensor logits and
        # requires TP world size > 1 to be meaningful.
        if self.tp_loss_parallel and self.fsdp2_tp_size <= 1:
            raise ValueError("--tp_loss_parallel requires --fsdp2_tp_size > 1.")

        # FSDP config
        self.param_dtype = args.param_dtype
        self.fsdp2_cpu_offload = args.fsdp2_cpu_offload
        self.fsdp2_reshard_after_forward = args.fsdp2_reshard_after_forward
        self.sequence_parallel = args.sequence_parallel

        # CPUOffloadPolicy and manual offload are mutually exclusive (ref: slime)
        # When fsdp2_cpu_offload is enabled, FSDP2 manages CPU offload automatically,
        # so fsdp2_enable_sleep (manual offload) should be disabled.
        if self.fsdp2_cpu_offload and getattr(args, "fsdp2_enable_sleep", False):
            args.fsdp2_enable_sleep = False
            print("[FSDP2] Warning: fsdp2_enable_sleep disabled because fsdp2_cpu_offload is enabled")

        # State
        self.time_steps = defaultdict(int)
        self.mesh = None
        self._gloo_group = None  # Gloo group for CPU-safe barriers

        # Derive checkpoint sub-paths from ckpt_save_path (training scripts only)
        ckpt_save_path = getattr(args, "ckpt_save_path", None)
        if ckpt_save_path is not None:
            args.last_hf_ckpt_path = os.path.join(ckpt_save_path, "last_hf_ckpt")
            args.hf_ckpt_path = os.path.join(ckpt_save_path, "hf_ckpt")
            args.dcp_ckpt_path = os.path.join(ckpt_save_path, "dcp_ckpt")

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

        # Initialize Gloo group for CPU-safe barriers (ref: slime)
        # NCCL barriers require GPU tensors, which fail when model is offloaded to CPU.
        # Gloo works with CPU tensors and is safe to use after model offload.
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
                raise ValueError("Invalid config: --sequence_parallel requires --fsdp2_tp_size > 1.")
            if not getattr(self.args, "packing_samples", False):
                raise ValueError(
                    "--sequence_parallel requires --packing_samples to be enabled, "
                    "because HF's causal mask creation uses inputs_embeds.shape[1] "
                    "which would be seq/tp_size after SP sharding, causing mask length mismatch."
                )

        # Create 3D mesh: (dp, cp, tp) - always include all dimensions even if size=1
        # This ensures all parameters use the same mesh structure, avoiding mesh mismatch issues
        # in operations like clip_grad_norm that aggregate across all parameters
        self.mesh = init_device_mesh(
            "cuda",
            (self.fsdp2_dp_size, self.fsdp2_cp_size, self.fsdp2_tp_size),
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
        set_ring_attn_pad_multiple(1)

        if self.sequence_parallel and self.fsdp2_tp_size > 1:
            # Ensure packed sequence length is divisible by TP degree when SP is enabled.
            # - If CP is enabled, this pads total length to `fsdp2_cp_size * fsdp2_tp_size`.
            # - If CP is disabled, this pads total length to `fsdp2_tp_size`.
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

        # Gradient accumulation
        # Only DP contributes to batch size - CP processes same sequence chunks, TP processes same batch
        # This matches the standard formula: train_batch_size / (micro_train_batch_size * fsdp2_dp_size)
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

        self._log(
            f"world={self.world_size} dp={self.fsdp2_dp_size} cp={self.fsdp2_cp_size} "
            f"tp={self.fsdp2_tp_size} grad_accum={self.accumulated_gradient} "
            f"fsdp_mesh_size={self.fsdp2_dp_size * self.fsdp2_cp_size}"
        )

        # TP-aware loss is handled by DTensor-aware helpers (no monkey patch needed).

    @property
    def ring_attn_group(self):
        return get_ring_attn_group()

    # -------------------------------------------------------------------------
    # Model Wrapping
    # -------------------------------------------------------------------------

    def apply_parallelism(self, model, force_cpu_offload: bool = False):
        """Apply TP + FSDP sharding to a model, preserving Actor wrapper.

        Args:
            model: Model to shard.
            force_cpu_offload: If True, enable CPU offload policy for this model.
        """
        unwrapped = self._unwrap_model(model)
        is_actor = unwrapped is not model

        # TP before FSDP
        self._log(
            f"Wrapping model with dp={self.fsdp2_dp_size} cp={self.fsdp2_cp_size} tp={self.fsdp2_tp_size} "
            f"(fsdp_mesh_size={self.fsdp2_dp_size * self.fsdp2_cp_size})"
        )
        if self.fsdp2_tp_size > 1:
            self._log(f"Applying TP (size={self.fsdp2_tp_size})")
            unwrapped = apply_tensor_parallel(
                unwrapped,
                self.mesh["tp"],
                sequence_parallel=self.sequence_parallel,
                validate=True,
                shard_logits=self.tp_loss_parallel,
            )
        else:
            self._log("Skipping TP (fsdp2_tp_size=1)")

        # FSDP (force_cpu_offload overrides self.fsdp2_cpu_offload)
        unwrapped = self._apply_fsdp(unwrapped, force_cpu_offload=force_cpu_offload)

        if is_actor:
            model.model = unwrapped
            return model
        return unwrapped

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

        # Shard transformer layers
        no_split_modules = getattr(model, "_no_split_modules", [])
        fsdp_units = [
            m
            for m in model.modules()
            if m.__class__.__name__ in no_split_modules
            or (
                isinstance(m, nn.Embedding)
                and not getattr(getattr(model, "config", None), "tie_word_embeddings", True)
            )
        ]

        for i, layer in enumerate(fsdp_units):
            if not isinstance(layer, FSDPModule):
                fully_shard(
                    layer,
                    mesh=mesh,
                    mp_policy=mixed_precision,
                    offload_policy=offload_policy,
                    reshard_after_forward=self.fsdp2_reshard_after_forward and i < len(fsdp_units) - 1,
                )

        # Shard root
        if not isinstance(model, FSDPModule):
            fully_shard(model, mesh=mesh, mp_policy=mixed_precision, offload_policy=offload_policy, reshard_after_forward=False)
        return model

    # -------------------------------------------------------------------------
    # Model Loading & Post-load Fixup
    # -------------------------------------------------------------------------

    def load_hf_checkpoint(
        self,
        model: nn.Module,
        pretrain: str,
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

        on_cpu = self.fsdp2_cpu_offload or force_cpu_offload
        device = "cpu" if on_cpu else torch.device("cuda", torch.cuda.current_device())

        _load_hf_checkpoint(
            unwrapped,
            pretrain,
            device=device,
            process_group=self._gloo_group,
        )

        if on_cpu:
            self._move_buffers_to_cuda(unwrapped)
        if init_value_head:
            self._init_value_head_after_load(unwrapped, value_head_prefix=value_head_prefix)
        return True

    @torch.no_grad()
    def _move_buffers_to_cuda(self, model: nn.Module) -> None:
        """Keep buffers on CUDA for CPU-offloaded FSDP2 models (slime-style)."""
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
            # Match existing init: std = 1 / (hidden_size + 1)
            hidden_size = shape[1] if len(shape) >= 2 else max(1, shape[0])
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
        grouped = self._get_optimizer_grouped_parameters(self._unwrap_model(model), weight_decay)
        # fused=True only works on CUDA and is incompatible with CPU offload.
        fused = torch.cuda.is_available() and not self.fsdp2_cpu_offload
        return optim.AdamW(grouped, fused=fused, **kwargs)

    @staticmethod
    def _get_optimizer_grouped_parameters(
        model: nn.Module,
        weight_decay: float,
        no_decay_name_list=("bias", "layer_norm.weight", "layernorm.weight", "norm.weight", "ln_f.weight"),
    ):
        """Match OpenRLHF's optimizer parameter grouping rules."""
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
        """Backward with gradient accumulation and deferred sync.

        Note on loss scaling with CP (Context Parallelism):
        - Ring Attention slices the sequence across CP ranks for the forward pass,
          then uses `flash_attn.utils.distributed.all_gather` to rebuild full-sequence tensors
          (log_probs / values / etc.) on EVERY CP rank before loss computation.
        - `all_gather` autograd uses reduce_scatter(SUM) in backward, which introduces a `fsdp2_cp_size`
          factor into the local gradients.
        - FSDP2 then averages gradients across the flattened `dp_cp` mesh, dividing by
          (fsdp2_dp_size * fsdp2_cp_size); the `fsdp2_cp_size` factor from `all_gather` cancels this, yielding an
          effective "dp average, cp sum" without explicitly scaling the loss here.

        MoE aux_loss handling:
        - Trainers combine main_loss and aux_loss before calling backward:
          loss = main_loss + aux_loss * aux_loss_coef
        - aux_loss is computed locally on each CP rank based on the tokens it processes.
        - FSDP2's AVG across (fsdp2_dp_size * fsdp2_cp_size) correctly averages aux_loss across all ranks,
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
            # Use DTensor-compatible gradient clipping for TP+FSDP
            # Standard clip_grad_norm_ fails when params have different meshes
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

        Note:
            Data is sharded only across DP dimension, not CP.
            CP ranks within the same DP group see the same data but process different
            sequence chunks. This is essential for Ring Attention to work correctly.
        """
        if sampler is None and dist.is_initialized():
            # Use dp_group for data sharding - CP ranks share the same data
            # This ensures all CP ranks in the same DP group see identical samples
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
        """Offload model params to CPU.

        Call this AFTER weight sync with vLLM to free GPU memory.
        This is used in the hybrid engine mode where training model is offloaded
        during rollout phase to give vLLM more GPU memory.
        """
        if self.fsdp2_cpu_offload:
            return  # CPUOffloadPolicy already manages model

        unwrapped = self._unwrap_model(model)
        unwrapped.cpu()
        if self.is_rank_0():
            print("[FSDP2] Model offloaded to CPU")
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def reload_model(self, model):
        """Reload model params to GPU.

        Call this BEFORE forward pass when model was offloaded.
        This is used in the hybrid engine mode to reload training model
        before computing log probs.
        """
        if self.fsdp2_cpu_offload:
            return  # CPUOffloadPolicy already manages model

        unwrapped = self._unwrap_model(model)
        if next(unwrapped.parameters()).device.type == "cpu":
            unwrapped.to(torch.device("cuda", torch.cuda.current_device()))
            torch.cuda.synchronize()
            if self.is_rank_0():
                print("[FSDP2] Model reloaded to GPU")

    @torch.no_grad()
    def offload_states(self, model, optimizer):
        """Offload optimizer states to CPU (model stays on GPU).

        Use this for vLLM weight sync: optimizer offloaded, model stays for sync.
        """
        move_optimizer_state(optimizer, torch.device("cpu"))
        if self.is_rank_0():
            print("[FSDP2] Optimizer states offloaded to CPU")

        torch.cuda.empty_cache()
        if self._gloo_group is not None:
            dist.barrier(group=self._gloo_group)
        torch.cuda.synchronize()

    @torch.no_grad()
    def reload_states(self, model, optimizer):
        """Reload optimizer states to GPU (model not included)."""
        device = torch.device("cuda", torch.cuda.current_device())
        move_optimizer_state(optimizer, device)
        if self.is_rank_0():
            print("[FSDP2] Optimizer states reloaded to GPU")

        torch.cuda.synchronize()
        if self._gloo_group is not None:
            dist.barrier(group=self._gloo_group)

    # -------------------------------------------------------------------------
    # Checkpointing
    # -------------------------------------------------------------------------

    def save_hf_checkpoint(self, model, tokenizer, output_dir, tag=None, max_num=None, max_mem=None, **kwargs):
        """Save HF checkpoint, optionally under a sub-tag with rotation cleanup.

        Args:
            model: Model to save.
            tokenizer: Tokenizer to save alongside the model.
            output_dir: Root directory for saving.
            tag: If provided, save under ``output_dir/tag`` and apply rotation cleanup.
            max_num: Max number of checkpoints to keep (used with tag).
            max_mem: Max total checkpoint size in GB (used with tag).
        """
        save_dir = os.path.join(output_dir, tag) if tag else output_dir

        # Convert fp32 master weights to the training compute dtype (e.g. bf16) for deployment.
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

        if tag and self.is_rank_0():
            os.makedirs(output_dir, exist_ok=True)
            _cleanup_old_checkpoints(output_dir, max_num, max_mem)

    def save_dcp_checkpoint(
        self,
        model,
        save_dir,
        tag=None,
        max_num=3,
        max_mem=1000,
        client_state=None,
        save_latest=True,
        optimizer=None,
        scheduler=None,
    ):
        """Save FSDP2 distributed checkpoint."""
        _save_dcp_checkpoint(
            model,
            save_dir,
            tag,
            self._unwrap_model,
            self.is_rank_0(),
            optimizer=optimizer,
            scheduler=scheduler,
            client_state=client_state,
            max_num=max_num,
            max_mem=max_mem,
            save_latest=save_latest,
            process_group=self._gloo_group,
        )

    def load_dcp_checkpoint(
        self,
        model,
        load_dir,
        tag=None,
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
            tag=tag,
            optimizer=optimizer,
            scheduler=scheduler,
            load_optimizer_states=load_optimizer_states,
            load_lr_scheduler_states=load_lr_scheduler_states,
            load_module_strict=load_module_strict,
            load_module_only=load_module_only,
            process_group=self._gloo_group,
        )

    # -------------------------------------------------------------------------
    # Communication
    # -------------------------------------------------------------------------

    def all_reduce(self, data, op="mean", with_context_parallel=True):
        """All-reduce across DP group (optionally including CP).

        Args:
            data: Data to reduce (tensor, scalar, or dict)
            op: Reduction operation ("mean", "max", "sum")
            with_context_parallel: If True, reduce across both DP and CP (for metrics).
                                   If False, reduce across DP only (for data partitioning).
        """
        assert op in ("mean", "max", "sum")

        if isinstance(data, dict):
            return {k: self.all_reduce(v, op, with_context_parallel) for k, v in data.items()}

        if not dist.is_initialized():
            return data

        # dp_cp_group for metrics, dp_group for data operations
        process_group = self.dp_cp_group if with_context_parallel and self.fsdp2_cp_size > 1 else self.dp_group
        group_size = dist.get_world_size(group=process_group)

        # Convert scalar to tensor
        was_tensor = isinstance(data, torch.Tensor)
        tensor = data if was_tensor else torch.tensor(data, device="cuda")

        # Move to GPU if needed (NCCL requires CUDA tensors)
        from_cpu = tensor.device.type == "cpu"
        if from_cpu:
            tensor = tensor.cuda()

        # Perform all-reduce
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

        # Convert scalar to tensor
        was_tensor = isinstance(data, torch.Tensor)
        tensor = data if was_tensor else torch.tensor(data, device="cuda")

        # Handle 0-dim tensors
        if tensor.dim() == 0:
            tensor = tensor.view(1)

        from_cpu = tensor.device.type == "cpu"
        gpu_tensor = tensor.cuda() if from_cpu else tensor

        # Gather
        shards = [torch.zeros_like(gpu_tensor) for _ in range(group_size)]
        dist.all_gather(shards, gpu_tensor, group=process_group)
        result = torch.cat(shards)

        if from_cpu:
            result = result.cpu()

        return result if was_tensor else result.tolist()

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

    def _unwrap_model(self, model):
        """Unwrap Actor wrapper to get the underlying module."""
        if isinstance(model, Actor):
            return self._unwrap_model(model.model)
        else:
            return model
