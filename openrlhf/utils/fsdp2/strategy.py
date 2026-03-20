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
from collections import defaultdict
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffloadPolicy, FSDPModule, MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import enable_full_determinism, set_seed

from openrlhf.models import Actor
from openrlhf.models.ring_attn_utils import get_ring_attn_group, set_ring_attn_group, set_ring_attn_pad_multiple
from openrlhf.utils import convert_to_torch_dtype
from openrlhf.utils.distributed_sampler import DistributedSampler

from .checkpoint import (
    _cleanup_old_checkpoints,
    _fixup_after_load,
    _load_dcp_checkpoint,
    _load_hf_checkpoint,
    _save_dcp_checkpoint,
    _save_hf_checkpoint,
)
from .tp.tp_parallel import apply_tensor_parallel
from .utils import (
    clip_grad_norm_dtensor,
    get_checkpoint_metadata,
    move_buffers_to_cuda,
    move_optimizer_state,
    moving_average_fsdp2,
)


class FSDP2Strategy:
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
        self._last_grad_norm_by_model = {}
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

        # 1. Process group
        if self.args.local_rank != -1:
            os.environ["LOCAL_RANK"] = str(self.args.local_rank)
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if local_rank >= 0:
            torch.cuda.set_device(local_rank)

        backend = "nccl" if torch.cuda.is_available() else "gloo"
        device_id = local_rank if (backend == "nccl" and local_rank >= 0) else None
        dist.init_process_group(backend=backend, timeout=timeout, device_id=device_id)
        self.world_size = dist.get_world_size()
        # Gloo group for CPU-safe barriers (NCCL requires GPU tensors)
        self._gloo_group = dist.new_group(backend="gloo")

        # 2. Device mesh (dp, cp, tp)
        cp_tp = self.fsdp2_cp_size * self.fsdp2_tp_size
        assert self.world_size % cp_tp == 0, f"world_size({self.world_size}) not divisible by cp*tp({cp_tp})"
        self.fsdp2_dp_size = self.world_size // cp_tp

        if self.sequence_parallel:
            if self.fsdp2_tp_size <= 1:
                raise ValueError("--fsdp2_tp_sequence_parallel requires --fsdp2_tp_size > 1.")
            if not getattr(self.args, "packing_samples", False):
                raise ValueError(
                    "--fsdp2_tp_sequence_parallel requires --packing_samples "
                    "(HF causal mask uses inputs_embeds.shape[1] which would be seq/tp after SP sharding)."
                )

        self.mesh = init_device_mesh(
            "cuda",
            (self.fsdp2_dp_size, self.fsdp2_cp_size, self.fsdp2_tp_size),
            mesh_dim_names=("dp", "cp", "tp"),
        )
        # Merge DP+CP for FSDP reduce-scatter (see module docstring)
        self.fsdp_mesh = self.mesh["dp", "cp"]._flatten(mesh_dim_name="dp_cp")

        self.dp_cp_group = self.fsdp_mesh.get_group()
        self.dp_group = self.mesh["dp"].get_group()
        self.cp_group = self.mesh["cp"].get_group()

        # 3. Ring Attention (context parallelism)
        set_ring_attn_group(None)
        set_ring_attn_pad_multiple(1)

        if self.sequence_parallel and self.fsdp2_tp_size > 1:
            set_ring_attn_pad_multiple(self.fsdp2_tp_size)

        if self.fsdp2_cp_size > 1:
            set_ring_attn_group(self.cp_group)
            try:
                from ring_flash_attn import substitute_hf_flash_attn
            except ModuleNotFoundError as e:
                raise RuntimeError(
                    "ring_flash_attn is required when --fsdp2_cp_size > 1. "
                    "Install ring_flash_attn or set --fsdp2_cp_size 1."
                ) from e
            substitute_hf_flash_attn(self.cp_group, getattr(self.args, "ring_head_stride", 1))

        # 4. Gradient accumulation
        if getattr(self.args, "use_dynamic_batch", False):
            self.accumulated_gradient = 1
        else:
            batch_per_step = self.micro_train_batch_size * self.fsdp2_dp_size
            accum_steps, remainder = divmod(self.train_batch_size, batch_per_step)
            if accum_steps < 1 or remainder != 0:
                raise ValueError(
                    f"Invalid batch config: train_batch_size({self.train_batch_size}) must equal "
                    f"micro_train_batch_size({self.micro_train_batch_size}) * dp({self.fsdp2_dp_size}) * N."
                )
            self.accumulated_gradient = accum_steps

        self.print(
            f"[FSDP2] world={self.world_size} dp={self.fsdp2_dp_size} cp={self.fsdp2_cp_size} "
            f"tp={self.fsdp2_tp_size} grad_accum={self.accumulated_gradient} "
            f"fsdp_mesh_size={self.fsdp2_dp_size * self.fsdp2_cp_size}"
        )

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
            fully_shard(
                model, mesh=mesh, mp_policy=mixed_precision, offload_policy=offload_policy, reshard_after_forward=False
            )
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
        init_missing_value_head: bool = False,
        force_init_value_head: bool = False,
    ) -> bool:
        """Load pretrained HF weights into an already-materialized model.

        The caller must have called ``model_to_empty`` before this method.

        Args:
            init_missing_value_head: If True and ``score.weight`` is missing
                from the checkpoint, randomly initialize it.  Use this when
                training a new reward/critic model from a base LM.  Inference
                callers must NOT set this — a missing score.weight means the
                checkpoint is wrong and should fail fast.
            force_init_value_head: Always reinitialize ``score.weight``, even
                if it exists in the checkpoint.

        Returns:
          True on success.
        """
        unwrapped = self._unwrap_model(model)

        missing_keys = set(
            _load_hf_checkpoint(
                unwrapped,
                model_name_or_path,
                process_group=self._gloo_group,
            )
        )

        # Value-head: only init score.weight when caller explicitly opts in.
        # Inference paths must NOT auto-init — a missing score.weight means a bad checkpoint.
        should_init = force_init_value_head or (init_missing_value_head and "score.weight" in missing_keys)
        if should_init:
            reason = "forced" if force_init_value_head else "missing from checkpoint"
            self.print(f"[FSDP2] Initializing score.weight ({reason})")
            self._init_value_head_after_load(unwrapped)
            missing_keys.discard("score.weight")

        if missing_keys:
            sample_keys = ", ".join(sorted(missing_keys)[:8])
            raise RuntimeError(
                "HF checkpoint is missing required keys after FSDP2 materialization: "
                f"{sample_keys}. Refusing to leave parameters uninitialized."
            )

        self._post_load_fixup(model, force_cpu_offload=force_cpu_offload)
        return True

    def _post_load_fixup(self, model: nn.Module, *, force_cpu_offload: bool = False) -> None:
        """Repair non-persistent state after to_empty() + weight load.

        Handles: tied word embeddings, rotary inv_freq, reward norm buffers.
        Called by both ``load_hf_checkpoint`` and ``load_dcp_resume``.
        """
        unwrapped = self._unwrap_model(model)
        _fixup_after_load(unwrapped)
        if self.fsdp2_cpu_offload or force_cpu_offload:
            move_buffers_to_cuda(unwrapped)

    @torch.no_grad()
    def _init_value_head_after_load(self, model: nn.Module) -> None:
        """Initialize score.weight after meta-init + checkpoint load.

        Generates random weights on rank 0, broadcasts to all ranks via set_model_state_dict.
        Used when training reward/critic from a base LM that lacks score.weight.
        """
        value_head = getattr(model, "score", None)
        if value_head is None or not hasattr(value_head, "weight"):
            raise RuntimeError("Reward/Critic model must expose `score.weight`.")

        weight = value_head.weight
        if dist.is_initialized() and dist.get_rank() != 0:
            state = {}
        else:
            shape = tuple(weight.shape)
            dtype = weight.dtype
            hidden_size = shape[1] if len(shape) >= 2 else max(1, shape[0])
            std = 1.0 / float(hidden_size + 1)
            init_weight = torch.empty(shape, device="cpu", dtype=dtype)
            init_weight.normal_(mean=0.0, std=std)
            state = {"score.weight": init_weight}

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

    def model_to_empty(self, model: nn.Module, *, force_cpu_offload: bool = False) -> None:
        """Materialize a meta-init model onto the target device without loading weights.

        Call this before creating the optimizer so that the optimizer binds to
        the real (materialized) parameter objects.  ``load_dcp_resume`` will
        then fill the values in-place via ``set_state_dict``.
        """
        unwrapped = self._unwrap_model(model)
        load_to_cpu = self.fsdp2_cpu_offload or force_cpu_offload
        device = "cpu" if load_to_cpu else torch.device("cuda", torch.cuda.current_device())
        unwrapped.to_empty(device=device)

    def load_dcp_resume(
        self,
        model: nn.Module,
        load_dir: str,
        *,
        force_cpu_offload: bool = False,
        load_module_only: bool = False,
        optimizer=None,
        scheduler=None,
    ) -> dict:
        """Load DCP checkpoint into an already-materialized model.

        The model must have been materialized (e.g. via ``model_to_empty``)
        before calling this, and the optimizer must have been created after
        materialization so that it references the correct parameter objects.

        Flow: DCP load (in-place) → fixup.
        Returns client_state dict.
        """
        client_state = _load_dcp_checkpoint(
            model,
            load_dir,
            self._unwrap_model,
            optimizer=optimizer,
            scheduler=scheduler,
            load_optimizer_states=not load_module_only,
            load_lr_scheduler_states=not load_module_only,
            load_module_only=load_module_only,
            process_group=self._gloo_group,
        )

        self._post_load_fixup(model, force_cpu_offload=force_cpu_offload)
        return client_state

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    def create_optimizer(self, model, **kwargs):
        """Create AdamW optimizer."""
        if "foreach" not in kwargs and self.fsdp2_tp_size > 1 and self.fsdp2_dp_size <= 1:
            kwargs["foreach"] = False
        weight_decay = kwargs.pop("weight_decay", 0.0)
        param_groups = self._get_optimizer_grouped_parameters(self._unwrap_model(model), weight_decay)
        fused = torch.cuda.is_available() and not self.fsdp2_cpu_offload
        return optim.AdamW(param_groups, fused=fused, **kwargs)

    @staticmethod
    def _get_optimizer_grouped_parameters(
        model: nn.Module,
        weight_decay: float,
        no_decay_name_list=("bias", "layer_norm.weight", "layernorm.weight", "norm.weight", "ln_f.weight"),
    ):
        """Separate parameters into weight-decay and no-weight-decay groups."""
        decay, no_decay = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            (no_decay if any(nd in name for nd in no_decay_name_list) else decay).append(param)
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    def _is_sync_step(self, key: str, sync_gradients=None, *, peek_next: bool = False) -> bool:
        """Determine whether the current (or next) step is a gradient sync step.

        Args:
            key: Time-step counter key (e.g. "step_actor", "ema").
            sync_gradients: Explicit override; if not None, used directly.
            peek_next: If True, check the *next* step count without incrementing.
        """
        if sync_gradients is not None:
            return bool(sync_gradients)
        step = self.time_steps.get(key, 0)
        if peek_next:
            step += 1
        return step % max(1, self.accumulated_gradient) == 0

    def backward(self, loss, model, optimizer, name="model", sync_gradients=None, **kwargs):
        """Backward pass with sharded gradient accumulation.

        FSDP2 gradient accumulation strategy:
        - FSDP2 always reduce-scatters gradients on every backward call, keeping
          each rank's .grad at O(params/world_size) memory.
        - Unlike DeepSpeed, we cannot disable gradient sync — FSDP2's reduce-scatter
          is needed to produce correctly sharded gradients.
        - For gradient accumulation (accumulated_gradient > 1), we manually buffer
          sharded .grad into per-parameter `_fsdp2_acc_grad` on non-sync steps,
          then restore them in optimizer_step() before the optimizer runs.
        - This achieves the same effect as DeepSpeed's no_sync(), but within
          FSDP2's always-sync design.
        """
        if self.accumulated_gradient > 1:
            loss = loss / self.accumulated_gradient

        unwrapped = self._unwrap_model(model)

        # Peek at whether the NEXT optimizer_step will be a sync step
        key = f"step_{name}"
        is_sync = self._is_sync_step(key, sync_gradients, peek_next=True)

        if isinstance(unwrapped, FSDPModule):
            unwrapped.set_requires_gradient_sync(True)

        loss.backward()

        # Buffer sharded grads on non-sync (accumulation) steps:
        # save .grad into per-param buffer and clear .grad so next micro-batch starts fresh
        if isinstance(unwrapped, FSDPModule) and not is_sync:
            for p in unwrapped.parameters():
                if p.grad is not None:
                    acc = getattr(p, "_fsdp2_acc_grad", None)
                    if acc is not None:
                        acc.add_(p.grad)
                    else:
                        p._fsdp2_acc_grad = p.grad.clone()
                    p.grad = None

    def optimizer_step(self, optimizer, model, scheduler, name="model", sync_gradients=None, **kwargs):
        """Optimizer step — only executes on sync steps (every accumulated_gradient steps)."""
        key = f"step_{name}"
        self.time_steps[key] += 1
        is_sync = self._is_sync_step(key, sync_gradients)
        if not is_sync:
            return

        # Restore accumulated sharded grads from previous micro-batches
        unwrapped = self._unwrap_model(model)
        if isinstance(unwrapped, FSDPModule):
            for p in unwrapped.parameters():
                acc = getattr(p, "_fsdp2_acc_grad", None)
                if acc is not None:
                    if p.grad is not None:
                        p.grad.add_(acc)
                    else:
                        p.grad = acc
                    del p._fsdp2_acc_grad

        # Cache pre-clip grad norm for callers that read it after optimizer_step
        # (e.g. SFT/DPO/RM trainers where get_grad_norm is called post-step).
        norm = self._compute_grad_norm(unwrapped)
        if norm is not None:
            self._last_grad_norm_by_model[id(unwrapped)] = norm

        if self.max_norm > 0:
            clip_grad_norm_dtensor(unwrapped, max_norm=self.max_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if scheduler:
            scheduler.step()

    def moving_average(self, model, model_ema, beta=0.992, device="cpu", sync_gradients=None):
        """Update EMA model — only on sync steps."""
        self.time_steps["ema"] += 1
        if not self._is_sync_step("ema", sync_gradients):
            return
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
        """All-reduce across DP (optionally DP+CP). Accepts tensor, scalar, or dict."""
        assert op in ("mean", "max", "sum")
        if isinstance(data, dict):
            return {k: self.all_reduce(v, op, with_context_parallel) for k, v in data.items()}
        if not dist.is_initialized():
            return data

        group = self.dp_cp_group if with_context_parallel and self.fsdp2_cp_size > 1 else self.dp_group

        was_tensor = isinstance(data, torch.Tensor)
        tensor = data if was_tensor else torch.tensor(data, device="cuda")
        from_cpu = tensor.device.type == "cpu"
        if from_cpu:
            tensor = tensor.cuda()

        reduce_op = {"mean": dist.ReduceOp.SUM, "max": dist.ReduceOp.MAX, "sum": dist.ReduceOp.SUM}[op]
        dist.all_reduce(tensor, op=reduce_op, group=group)
        if op == "mean":
            tensor /= dist.get_world_size(group=group)

        if from_cpu:
            tensor = tensor.cpu()
        return tensor if was_tensor else tensor.item()

    def all_gather(self, data):
        """All-gather across DP (not CP/TP — they share the same data)."""
        if isinstance(data, dict):
            return {k: self.all_gather(v) for k, v in data.items()}
        if not dist.is_initialized():
            return data

        group = self.dp_group

        was_tensor = isinstance(data, torch.Tensor)
        tensor = data if was_tensor else torch.tensor(data, device="cuda")
        from_cpu = tensor.device.type == "cpu"
        if from_cpu:
            tensor = tensor.cuda()
        if tensor.dim() == 0:
            tensor = tensor.view(1)

        shards = [torch.zeros_like(tensor) for _ in range(dist.get_world_size(group=group))]
        dist.all_gather(shards, tensor, group=group)
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

    def get_grad_norm(self, model) -> float:
        """Return current grad norm, including FSDP2 accumulation buffers.

        If grads were already cleared by optimizer_step(), fall back to the most
        recently cached pre-step value so trainer-side logging remains meaningful.
        """
        unwrapped = self._unwrap_model(model)
        live_norm = self._compute_grad_norm(unwrapped)
        if live_norm is not None:
            self._last_grad_norm_by_model[id(unwrapped)] = live_norm
            return live_norm
        return self._last_grad_norm_by_model.get(id(unwrapped), 0.0)

    @staticmethod
    def _compute_grad_norm(model: nn.Module) -> float | None:
        """Compute total norm from live grads and FSDP2 accumulation buffers."""
        grads = []
        for p in model.parameters():
            grad = p.grad
            acc = getattr(p, "_fsdp2_acc_grad", None)
            if grad is None and acc is None:
                continue
            if grad is None:
                grads.append(acc.detach())
            elif acc is None:
                grads.append(grad.detach())
            else:
                grads.append((grad.detach() + acc.detach()))

        if not grads:
            return None

        total_norm = torch.nn.utils.get_total_norm(grads, norm_type=2.0, error_if_nonfinite=False)
        if isinstance(total_norm, DTensor):
            if total_norm.to_local().device.type == "cpu" and torch.cuda.is_available():
                total_norm = total_norm.to(device=torch.device("cuda", torch.cuda.current_device()))
            total_norm = total_norm.full_tensor()
        return total_norm.detach().float().cpu().item()

    def _unwrap_model(self, model):
        """Unwrap Actor wrapper to get the underlying module."""
        if isinstance(model, Actor):
            return self._unwrap_model(model.model)
        return model
