import gc
import math
import os
from collections import defaultdict
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import transformers
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy
from torch.distributed.tensor import DTensor
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers.optimization import get_scheduler

from openrlhf.utils.distributed_sampler import DistributedSampler

try:
    from torch.distributed.fsdp._fully_shard import FSDPModule
except ImportError:  # pragma: no cover - torch version guard
    FSDPModule = None


def _get_actor_cls():
    """Lazy import to avoid circular dep: openrlhf.models.actor imports from this package."""
    from openrlhf.models import Actor

    return Actor


class FsdpStrategy:
    """FSDP2 + TP/CP/SP/EP backend using NeMo AutoModel.

    Mirrors DeepspeedStrategy's public surface so trainers are agnostic to the
    backend. The model is built and parallelized via AutoModel's official
    entry point ``NeMoAutoModelForCausalLM.from_pretrained`` inside ``Actor``;
    this strategy only handles distributed setup, optimizer/scheduler
    construction, the train-step (loss backward, grad clip, optimizer step),
    collectives, and checkpointing. Grad norm / clip are imported directly
    from ``nemo_automodel.components.distributed.grad_utils``.
    """

    CKPT_METRIC_FILENAME = "metric.json"
    _JSON_STATE_TYPE_KEY = "__openrlhf_type__"

    def __init__(
        self,
        seed: int = 42,
        full_determinism: bool = False,
        max_norm: float = 1.0,
        micro_train_batch_size: int = 1,
        train_batch_size: int = 1,
        args=None,
    ) -> None:
        self.args = args
        self.train_batch_size = train_batch_size
        self.micro_train_batch_size = micro_train_batch_size
        self.seed = seed
        self.full_determinism = full_determinism
        self.max_norm = max_norm

        fsdp = args.fsdp
        self.tp_size = getattr(fsdp, "tp_size", 1)
        self.cp_size = getattr(fsdp, "cp_size", 1)
        self.ep_size = getattr(fsdp, "ep_size", 1)
        self.pp_size = getattr(fsdp, "pp_size", 1)
        self.param_dtype = getattr(fsdp, "param_dtype", "bf16")
        self.cpu_offload = getattr(fsdp, "cpu_offload", False)
        sp = getattr(fsdp, "sequence_parallel", None)
        # SP defaults to ON whenever TP>1 (user-stated default).
        self.sequence_parallel = sp if sp is not None else (self.tp_size > 1)
        # LoRA flag: flows into AutoModel Checkpointer's `is_peft`. Without it
        # the checkpointer saves the merged base only and the trained adapter
        # is lost. Read the rank from the FSDP namespace if present.
        _lora_ns = getattr(fsdp, "lora", None)
        self.is_peft = getattr(_lora_ns, "rank", 0) > 0 if _lora_ns is not None else False

        self.world_size: int = 1
        self.device_mesh = None
        self.moe_mesh = None
        self.dp_size = 1
        self.dp_cp_size = 1
        self.accumulated_gradient: int = 1
        self._last_grad_norm: float = 0.0
        self.time_steps = defaultdict(int)
        self._max_norm_by_optimizer = {}

    # ProcessGroup / DeviceMesh aren't picklable. `datasets.map(self.process_data,
    # num_proc>1)` indirectly pickles the strategy via the dataset's bound method;
    # drop the distributed handles so the workers spawn cleanly. Workers don't need
    # them; they only run pure-CPU data preprocessing.
    _UNPICKLABLE_ATTRS = ("device_mesh", "moe_mesh", "distributed_config")

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k not in self._UNPICKLABLE_ATTRS}

    def __setstate__(self, state):
        self.__dict__.update(state)
        for k in self._UNPICKLABLE_ATTRS:
            self.__dict__.setdefault(k, None)

    def _get_automodel_mesh(self, name: str, required: bool = False):
        if self.device_mesh is None:
            return None

        try:
            try:
                from nemo_automodel.components.distributed.mesh_utils import get_flat_mesh
            except ImportError:
                get_flat_mesh = None

            if get_flat_mesh is not None:
                return get_flat_mesh(self.device_mesh, name)
            return self.device_mesh[name]
        except (KeyError, RuntimeError, AttributeError):
            if required:
                raise
            return None

    def _get_automodel_group(self, name: str):
        mesh = self._get_automodel_mesh(name, required=self.device_mesh is not None)
        return mesh.get_group() if mesh is not None else None

    def _get_dp_group(self, include_cp: bool = False):
        name = "dp_cp" if include_cp and self.cp_size > 1 else "dp"
        return self._get_automodel_group(name)

    def _get_dp_group_size(self, include_cp: bool = False) -> int:
        group = self._get_dp_group(include_cp=include_cp)
        if group is None:
            return dist.get_world_size() if dist.is_initialized() else 1
        return dist.get_world_size(group=group)

    def _get_automodel_rank(self, name: str) -> int:
        mesh = self._get_automodel_mesh(name, required=self.device_mesh is not None)
        return mesh.get_local_rank() if mesh is not None else 0

    def _get_dp_rank(self, include_cp: bool = False) -> int:
        if include_cp and self.cp_size > 1:
            return self._get_automodel_rank("dp_cp")
        return self._get_automodel_rank("dp")

    # ---------------------------------------------------------------- bring-up

    def setup_distributed(self, timeout: timedelta = timedelta(minutes=60)) -> None:
        if self.full_determinism:
            transformers.enable_full_determinism(self.seed)
        else:
            transformers.set_seed(self.seed)

        if self.args.local_rank != -1:
            os.environ["LOCAL_RANK"] = str(self.args.local_rank)

        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        if local_rank != -1:
            torch.cuda.set_device(local_rank)

        if not dist.is_initialized():
            backend = "cuda:nccl,cpu:gloo" if self.cpu_offload else "nccl"
            dist.init_process_group(backend=backend, timeout=timeout)

        self.world_size = dist.get_world_size()
        if self.world_size == 1 and self.cpu_offload:
            raise NotImplementedError(
                "CPU offload is not supported by AutoModel/FSDP2 on a single rank; "
                "disable --fsdp.cpu_offload or launch with more than one rank."
            )
        if self.pp_size > 1:
            raise NotImplementedError("OpenRLHF trainers are not pipeline-parallel aware yet; set --fsdp.pp_size 1")

        from nemo_automodel.components.distributed.config import FSDP2Config
        from nemo_automodel.components.distributed.mesh_utils import create_device_mesh
        from nemo_automodel.components.moe.config import MoEParallelizerConfig

        # Allow DPO/actor+ref TP embedding calls to reuse equivalent vocab masks.
        if self.tp_size > 1:
            try:
                from torch.distributed.tensor._ops import _mask_buffer
            except ImportError:
                _mask_buffer = None
            if _mask_buffer is not None and not getattr(_mask_buffer.MaskBuffer, "_orlhf_patched", False):

                def _safe_materialize(self, mask):
                    self.data = mask
                    self.refcount += 1

                _mask_buffer.MaskBuffer.materialize_mask = _safe_materialize
                _mask_buffer.MaskBuffer._orlhf_patched = True

        torch_dtype = _torch_dtype(self.param_dtype)
        # Match the FSDP2 baseline: params/forward in the requested dtype and
        # reduce-scatter in fp32. Do not force module outputs to fp32 here,
        # because policy/value losses should see the same dtype behavior as
        # the DeepSpeed and PR #1176 paths.
        mp_policy = (
            None
            if torch_dtype == torch.float32
            else MixedPrecisionPolicy(
                param_dtype=torch_dtype,
                reduce_dtype=torch.float32,
                cast_forward_inputs=True,
            )
        )
        # Public attribute: `Actor` reads it as `strategy.distributed_config`
        # and forwards to `NeMoAutoModelForCausalLM.from_pretrained`.
        # `activation_checkpointing` is intentionally NOT set here; Actor /
        # get_llm_for_sequence_regression pass it explicitly to from_pretrained
        # so the CLI flag (--model.gradient_checkpointing_enable /
        # --actor.gradient_checkpointing_enable) is the single source of truth.
        self.distributed_config = FSDP2Config(
            sequence_parallel=self.sequence_parallel,
            mp_policy=mp_policy,
            offload_policy=CPUOffloadPolicy(pin_memory=False) if self.cpu_offload else None,
            # Keep final microbatch grads materialized for clipping and logging.
            # Intermediate accumulation steps still disable sync in backward().
            defer_fsdp_grad_sync=False,
        )
        # MoE-specific parallelization config: required by AutoModel when
        # ep_size > 1 (raises 'NoneType has no to_dict' otherwise).
        self.moe_config = MoEParallelizerConfig(mp_policy=mp_policy) if self.ep_size > 1 else None

        self.device_mesh, self.moe_mesh = create_device_mesh(
            self.distributed_config,
            tp_size=self.tp_size,
            pp_size=self.pp_size,
            cp_size=self.cp_size,
            ep_size=self.ep_size,
            world_size=self.world_size,
        )

        # AutoModel's FSDP2 mesh exposes flattened "dp" for data loading and
        # "dp_cp" for FSDP reduce-scatter. Gradient accumulation is based on
        # DP only; CP ranks share samples and split sequence work.
        dp_size = self._get_dp_group_size(include_cp=False)
        self.dp_cp_size = self._get_dp_group_size(include_cp=True)
        if getattr(getattr(self.args, "train", None), "dynamic_batch_enable", False):
            self.accumulated_gradient = 1
        else:
            batch_per_step = self.micro_train_batch_size * dp_size
            accum_steps, remainder = divmod(self.train_batch_size, batch_per_step)
            if accum_steps < 1 or remainder != 0:
                raise ValueError(
                    "Invalid batch config for AutoModel/FSDP2: require "
                    "`train.batch_size = train.micro_batch_size * dp_size * grad_accum_steps` "
                    f"(got train.batch_size={self.train_batch_size}, "
                    f"train.micro_batch_size={self.micro_train_batch_size}, dp_size={dp_size})."
                )
            self.accumulated_gradient = accum_steps
        self.dp_size = dp_size
        self.print(
            f"[FSDP] world={self.world_size} dp={self.dp_size} cp={self.cp_size} tp={self.tp_size} "
            f"ep={self.ep_size} dp_cp={self.dp_cp_size} grad_accum={self.accumulated_gradient}"
        )

    # ---------------------------------------------------------------- prepare

    def prepare(self, *args):
        ret = []
        for arg in args:
            if isinstance(arg, tuple):
                assert len(arg) == 2, f"prepare() tuple must be (model, cfg); got len={len(arg)}"
                model, cfg = arg
                ret.append(self._init_train_model(model, cfg))
            else:
                ret.append(self._init_eval_model(arg))
        return ret[0] if len(ret) == 1 else ret

    def _init_train_model(self, model, cfg: dict):
        train_model = self._unwrap_model(model)
        params = [p for p in train_model.parameters() if p.requires_grad]
        if not params:
            raise ValueError("Cannot build optimizer: model has no trainable parameters")

        kind = cfg["optim"]
        adam = cfg["adam"]
        if kind == "muon":
            from openrlhf.utils.fsdp.muon import build_automodel_muon_optimizer

            optimizer = build_automodel_muon_optimizer(train_model, cfg["muon"], adam, self.device_mesh)
        elif kind == "adam":
            optimizer = torch.optim.AdamW(
                params,
                lr=adam["lr"],
                betas=tuple(adam["betas"]),
                eps=adam["eps"],
                weight_decay=adam["weight_decay"],
                foreach=False,
                fused=False,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {kind}")
        self._max_norm_by_optimizer[id(optimizer)] = cfg.get("max_norm", self.max_norm)

        scheduler_steps = cfg["scheduler_steps"]
        scheduler = get_scheduler(
            cfg.get("lr_scheduler", "cosine_with_min_lr"),
            optimizer,
            num_warmup_steps=math.ceil(scheduler_steps * cfg.get("lr_warmup_ratio", 0.03)),
            num_training_steps=scheduler_steps,
            scheduler_specific_kwargs={"min_lr_rate": cfg.get("min_lr_ratio", 0.1)},
        )
        return model, optimizer, scheduler

    def _init_eval_model(self, model):
        return model

    # ---------------------------------------------------------------- step loop

    @staticmethod
    def _set_fsdp_backward_sync(model: nn.Module, sync: bool) -> None:
        if FSDPModule is None:
            return
        fsdp_modules = [module for module in model.modules() if isinstance(module, FSDPModule)]
        if not fsdp_modules:
            return
        # Keep FSDP2 in its normal per-microbatch sync/reshard mode. NeMo-RL's
        # AutoModel path relies on loss scaling for accumulation instead of
        # holding unsharded FSDP params across the whole accumulation window;
        # the latter is especially expensive for CP/EP where weights are not
        # tensor-parallel sharded.
        fsdp_modules[0].set_is_last_backward(sync)
        fsdp_modules[0].set_reshard_after_backward(sync)
        fsdp_modules[0].set_requires_gradient_sync(sync)

    def backward(
        self,
        loss: torch.Tensor,
        model: nn.Module,
        optimizer: optim.Optimizer,
        name: str = "model",
        accumulate: bool = True,
        **kwargs,
    ) -> None:
        unwrapped = self._unwrap_model(model)
        if accumulate and self.accumulated_gradient > 1:
            if kwargs.get("scale_loss_by_accumulation", True):
                loss = loss / self.accumulated_gradient
        sync_gradients = kwargs.get("sync_gradients", True)
        self._set_fsdp_backward_sync(unwrapped, sync_gradients)
        if self.moe_mesh is not None:
            try:
                from nemo_automodel.components.moe.megatron.moe_utils import MoEAuxLossAutoScaler
            except ImportError:
                if self.is_rank_0():
                    print("[MoE] MoEAuxLossAutoScaler import failed; aux-loss scaling skipped.")
            else:
                MoEAuxLossAutoScaler.main_loss_backward_scale = torch.tensor(
                    float(getattr(self, "dp_cp_size", self.dp_size)),
                    device=loss.device,
                )
        loss.backward()

    def optimizer_step(
        self,
        optimizer: optim.Optimizer,
        model: nn.Module,
        scheduler,
        name: str = "model",
        accumulate: bool = True,
        **kwargs,
    ) -> None:
        # Skip the optimizer step until the last micro-batch in the accum window.
        key = f"step_{name}"
        if accumulate:
            self.time_steps[key] += 1
            if self.time_steps[key] % self.accumulated_gradient != 0:
                return

        model = self._unwrap_model(model)
        params = [p for p in model.parameters() if p.grad is not None]
        if not params:
            self._last_grad_norm = 0.0
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            return
        self._maybe_debug_grad_stats(model, name)
        max_norm = self._max_norm_by_optimizer.get(id(optimizer), self.max_norm)
        clip_norm = max_norm if max_norm and max_norm > 0 else None
        if clip_norm is not None or self.moe_mesh is not None:
            from nemo_automodel.components.training.utils import scale_grads_and_clip_grad_norm

            self._last_grad_norm = float(
                scale_grads_and_clip_grad_norm(
                    clip_norm,
                    [model],
                    pp_enabled=False,
                    device_mesh=self.device_mesh,
                    moe_mesh=self.moe_mesh,
                    ep_axis_name="ep" if self.moe_mesh is not None and "ep" in self.moe_mesh.mesh_dim_names else None,
                    foreach=False,
                    num_label_tokens=None,
                    dp_group_size=getattr(self, "dp_cp_size", self.dp_size),
                )
            )
        else:
            self._last_grad_norm = 0.0
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    def get_grad_norm(self, model: nn.Module) -> float:
        return self._last_grad_norm

    @staticmethod
    def _local_grad_tensor(grad: torch.Tensor) -> torch.Tensor:
        return grad.to_local() if isinstance(grad, DTensor) else grad

    def _maybe_debug_grad_stats(self, model: nn.Module, optim_name: str) -> None:
        debug = os.environ.get("OPENRLHF_FSDP_DEBUG_GRADS", "")
        if not debug or debug == "0":
            return
        enabled = {part.strip() for part in debug.split(",") if part.strip()}
        if "1" not in enabled and "all" not in enabled and optim_name not in enabled:
            return

        rank = dist.get_rank() if dist.is_initialized() else 0
        top_k = int(os.environ.get("OPENRLHF_FSDP_DEBUG_GRADS_TOPK", "8"))
        pattern_env = os.environ.get("OPENRLHF_FSDP_DEBUG_GRADS_FILTER")
        patterns = (
            [part.strip() for part in pattern_env.split(",") if part.strip()]
            if pattern_env
            else ["score", "lm_head", "embed_tokens", "layers.0.", "layers.31.", ".norm"]
        )
        rows = []
        nonfinite_tensors = 0
        total_tensors = 0
        total_elems = 0
        nonfinite_elems = 0
        total_sum_sq = 0.0

        for param_name, param in model.named_parameters():
            grad = param.grad
            if grad is None:
                continue
            if patterns and not any(pattern in param_name for pattern in patterns):
                continue
            total_tensors += 1
            local_grad = self._local_grad_tensor(grad).detach()
            total_elems += local_grad.numel()
            finite = torch.isfinite(local_grad)
            bad = local_grad.numel() - int(finite.sum().item())
            nonfinite_elems += bad
            if bad:
                nonfinite_tensors += 1
            finite_grad = torch.where(finite, local_grad, torch.zeros_like(local_grad)).double()
            local_sum_sq = finite_grad.pow(2).sum().item()
            total_sum_sq += local_sum_sq
            local_norm = math.sqrt(local_sum_sq)
            local_max = finite_grad.abs().max().item() if finite_grad.numel() else 0.0
            placement = tuple(str(p) for p in grad.placements) if isinstance(grad, DTensor) else ("local",)
            rows.append((local_norm, local_max, bad, param_name, placement, tuple(local_grad.shape)))

        rows.sort(key=lambda item: item[0], reverse=True)
        header = (
            f"[FSDPGradDebug][rank={rank}][{optim_name}] tensors={total_tensors} "
            f"nonfinite_tensors={nonfinite_tensors} elems={total_elems} nonfinite_elems={nonfinite_elems} "
            f"local_norm64={math.sqrt(total_sum_sq):.6e}"
        )
        print(header, flush=True)
        for local_norm, local_max, bad, param_name, placement, shape in rows[:top_k]:
            print(
                f"[FSDPGradDebug][rank={rank}][{optim_name}] "
                f"norm={local_norm:.6e} max={local_max:.6e} bad={bad} "
                f"shape={shape} placement={placement} name={param_name}",
                flush=True,
            )

    def global_token_count(self, mask: torch.Tensor) -> torch.Tensor:
        """All-reduce ``mask.sum()`` across the data-parallel data mesh.

        For CP, call this before ``make_cp_batch_and_ctx`` while each CP rank
        still sees the full local sequence. Token denominators are reduced over
        DP only because CP ranks share samples.
        """
        local = mask if mask.ndim == 0 else mask.sum()
        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else local.device
        local = local.to(dtype=torch.float32, device=device)
        dp_group = self._get_dp_group(include_cp=False)
        if dist.is_initialized() and dp_group is not None:
            dist.all_reduce(local, op=dist.ReduceOp.SUM, group=dp_group)
        return local

    # ---------------------------------------------------------------- data

    def setup_dataloader(
        self,
        replay_buffer,
        batch_size: int,
        pin_memory: bool = False,
        shuffle: bool = True,
        collate_fn=None,
        drop_last: bool = True,
        sampler=None,
        consumed_samples: int = 0,
        num_workers: int = 0,
    ):
        dp_group = self._get_dp_group(include_cp=False)
        if sampler is None and dist.is_initialized() and dp_group is not None:
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
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )

    # ---------------------------------------------------------------- comm

    def all_reduce(self, data, op: str = "mean"):
        if isinstance(data, dict):
            return {k: self.all_reduce(v, op) for k, v in data.items()}
        if not torch.is_tensor(data):
            data = torch.tensor(data, device=torch.cuda.current_device(), dtype=torch.float32)
        else:
            data = data.detach().clone().to(torch.cuda.current_device())
        reduce_op = {"mean": dist.ReduceOp.SUM, "sum": dist.ReduceOp.SUM, "max": dist.ReduceOp.MAX}[op]
        dist.all_reduce(data, op=reduce_op)
        if op == "mean":
            data = data / dist.get_world_size()
        return data.item() if data.ndim == 0 else data

    def all_gather(self, data):
        if not torch.is_tensor(data):
            data = torch.tensor(data, device=torch.cuda.current_device())
        gathered = [torch.zeros_like(data) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, data)
        return torch.cat(gathered, dim=0)

    def print(self, *msg):
        if self.is_rank_0():
            print(*msg)

    def is_rank_0(self) -> bool:
        return (not dist.is_initialized()) or dist.get_rank() == 0

    def get_rank(self) -> int:
        return dist.get_rank() if dist.is_initialized() else 0

    def _unwrap_model(self, model) -> nn.Module:
        if isinstance(model, _get_actor_cls()):
            return self._unwrap_model(model.model)
        if hasattr(model, "get_base_model_for_fsdp"):
            return model.get_base_model_for_fsdp()
        if hasattr(model, "module"):
            return model.module
        return model

    # ---------------------------------------------------------------- persistent offload
    #
    # Phase-boundary "sleep mode" for hybrid colocate (trainer + vLLM share
    # GPUs). Orthogonal to FSDP2's CPUOffloadPolicy: that one streams params
    # per-layer during compute, while these helpers bulk-move the whole model
    # between phases so vLLM can wake_up its weights+KV.

    @staticmethod
    def _move_buffers(model: nn.Module, device) -> None:
        # FSDP2 modules don't auto-move buffers via .to(); swap_tensors is
        # required so the storage actually moves and the old GPU storage is
        # released back to the allocator.
        for buf in model.buffers():
            torch.utils.swap_tensors(buf, buf.to(device))

    @staticmethod
    def _reshard_fsdp_modules(model: nn.Module) -> None:
        if FSDPModule is None:
            return
        for module in reversed(list(model.modules())):
            if not isinstance(module, FSDPModule):
                continue
            state = module._get_fsdp_state()
            fsdp_param_group = getattr(state, "_fsdp_param_group", None)
            if fsdp_param_group is None:
                continue
            try:
                module.reshard()
            except (AssertionError, RuntimeError):
                pass
            if not getattr(fsdp_param_group, "is_sharded", True):
                # If a forward raised before FSDP's post-forward hook, the
                # param group can still be in FORWARD with reshard_after_forward
                # disabled. Public reshard() is a no-op in that state, so force
                # the same sharded registration that reshard() would use from
                # IDLE before calling Module.to().
                to_sharded = getattr(fsdp_param_group, "_to_sharded", None)
                if to_sharded is None:
                    raise RuntimeError("FSDP param group cannot be resharded before moving devices")
                to_sharded()
            if not getattr(fsdp_param_group, "is_sharded", True):
                raise RuntimeError("FSDP param group stayed unsharded before moving devices")

    def move_model_to_device(self, model: nn.Module, device) -> None:
        """Bulk-move a model's params + buffers to ``device`` (cpu or cuda).
        Releases freed GPU storage to the caching allocator on exit."""
        unwrapped = self._unwrap_model(model)
        if str(device).startswith("cpu"):
            self._reshard_fsdp_modules(unwrapped)
        self._move_buffers(unwrapped, device)
        unwrapped.to(device)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def move_optimizer_to_device(optimizer: optim.Optimizer | None, device) -> None:
        """Bulk-move optimizer state tensors (Adam moments, step, etc.) to
        ``device``. ``model.to()`` does NOT touch these; they live on the
        optimizer, sharded as DTensors when params are DTensors."""
        if optimizer is None:
            return
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, (DTensor, torch.Tensor)):
                    state[k] = v.to(device)

    # ---------------------------------------------------------------- I/O

    @staticmethod
    def _checkpoint_source(model: nn.Module) -> tuple[str | None, str | None]:
        config = getattr(model, "config", None)
        model_repo_id = (
            getattr(model, "name_or_path", None)
            or getattr(config, "name_or_path", None)
            or getattr(config, "_name_or_path", None)
        )
        model_cache_dir = os.environ.get("HF_HUB_CACHE")
        if not model_cache_dir and os.environ.get("HF_HOME"):
            model_cache_dir = os.path.join(os.environ["HF_HOME"], "hub")
        return model_cache_dir, model_repo_id

    def save_model(self, model: nn.Module, tokenizer, output_dir: str, **kwargs) -> None:
        # Use AutoModel's Checkpointer: its custom-model save_pretrained mixin
        # requires it (raises "No checkpointer provided" otherwise). Outputs
        # consolidated HF safetensors that vLLM can hot-load.
        from nemo_automodel.components.checkpoint.checkpointing import Checkpointer, CheckpointingConfig

        # Pull peft_config off the wrapper *before* unwrap (Actor stashes it on
        # itself, not on .model). AutoModel's PEFT save addon needs the original
        # config to write adapter_config.json.
        peft_config = getattr(model, "peft_config", None)
        model = self._unwrap_model(model)
        model_cache_dir, model_repo_id = self._checkpoint_source(model)

        os.makedirs(output_dir, exist_ok=True)
        config = CheckpointingConfig(
            enabled=True,
            checkpoint_dir=output_dir,
            model_save_format="safetensors",
            model_cache_dir=model_cache_dir,
            model_repo_id=model_repo_id,
            save_consolidated=True,
            is_peft=self.is_peft,
            original_model_root_dir=model_cache_dir,
        )
        ckpt = Checkpointer(
            config=config,
            dp_rank=self._get_dp_rank(include_cp=True),
            tp_rank=self._get_automodel_rank("tp"),
            pp_rank=0,
            moe_mesh=self.moe_mesh,
        )
        ckpt.save_model(model=model, weights_path=output_dir, tokenizer=tokenizer, peft_config=peft_config)
        if dist.is_initialized():
            dist.barrier()
        self._promote_hf_export(output_dir)
        if dist.is_initialized():
            dist.barrier()

    @staticmethod
    def _promote_hf_export(output_dir: str) -> None:
        """Move AutoModel's HF export to ``output_dir`` for OpenRLHF callers."""
        if dist.is_initialized() and dist.get_rank() != 0:
            return

        import shutil

        model_dir = os.path.join(output_dir, "model")
        export_dir = os.path.join(model_dir, "consolidated")
        if not os.path.isdir(export_dir):
            export_dir = model_dir
        if not os.path.isdir(export_dir):
            return

        for name in os.listdir(export_dir):
            src = os.path.join(export_dir, name)
            dst = os.path.join(output_dir, name)
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            elif os.path.exists(dst):
                os.remove(dst)
            shutil.move(src, dst)
        shutil.rmtree(model_dir, ignore_errors=True)

    def _build_checkpointer(self, output_dir: str, save_consolidated: bool, model: nn.Module | None = None):
        from nemo_automodel.components.checkpoint.checkpointing import Checkpointer, CheckpointingConfig

        model_cache_dir, model_repo_id = self._checkpoint_source(model) if model is not None else (None, None)
        os.makedirs(output_dir, exist_ok=True)
        config = CheckpointingConfig(
            enabled=True,
            checkpoint_dir=output_dir,
            model_save_format="safetensors",
            model_cache_dir=model_cache_dir,
            model_repo_id=model_repo_id,
            save_consolidated=save_consolidated,
            is_peft=self.is_peft,
            original_model_root_dir=model_cache_dir,
        )
        return Checkpointer(
            config=config,
            dp_rank=self._get_dp_rank(include_cp=True),
            tp_rank=self._get_automodel_rank("tp"),
            pp_rank=0,
            moe_mesh=self.moe_mesh,
        )

    def _get_ckpt_metric_path(self, ckpt_dir: str) -> str:
        return os.path.join(ckpt_dir, self.CKPT_METRIC_FILENAME)

    @staticmethod
    def _atomic_write_text(path: str, text: str) -> None:
        tmp_path = f"{path}.tmp.{os.getpid()}"
        with open(tmp_path, "w") as f:
            f.write(text)
        os.replace(tmp_path, path)

    @staticmethod
    def _json_safe(obj):
        type_key = FsdpStrategy._JSON_STATE_TYPE_KEY
        if isinstance(obj, torch.Tensor):
            tensor = obj.detach().cpu()
            return {
                type_key: "tensor",
                "dtype": str(tensor.dtype).replace("torch.", ""),
                "shape": list(tensor.shape),
                "data": tensor.tolist(),
            }
        if isinstance(obj, torch.dtype):
            return {type_key: "dtype", "value": str(obj).replace("torch.", "")}
        if isinstance(obj, os.PathLike):
            return os.fspath(obj)
        if isinstance(obj, tuple):
            return {type_key: "tuple", "items": [FsdpStrategy._json_safe(v) for v in obj]}
        if isinstance(obj, list):
            return [FsdpStrategy._json_safe(v) for v in obj]
        if isinstance(obj, dict):
            if all(isinstance(k, str) for k in obj):
                return {k: FsdpStrategy._json_safe(v) for k, v in obj.items()}
            return {
                type_key: "dict",
                "items": [[FsdpStrategy._json_safe(k), FsdpStrategy._json_safe(v)] for k, v in obj.items()],
            }
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        if hasattr(obj, "item"):
            try:
                return obj.item()
            except Exception:
                pass
        return repr(obj)

    @staticmethod
    def _json_restore(obj):
        type_key = FsdpStrategy._JSON_STATE_TYPE_KEY
        if isinstance(obj, list):
            return [FsdpStrategy._json_restore(v) for v in obj]
        if not isinstance(obj, dict):
            return obj
        state_type = obj.get(type_key)
        if state_type == "tensor":
            dtype = getattr(torch, obj["dtype"])
            tensor = torch.tensor(obj["data"], dtype=dtype)
            return tensor.reshape(obj["shape"])
        if state_type == "dtype":
            return getattr(torch, obj["value"])
        if state_type == "tuple":
            return tuple(FsdpStrategy._json_restore(v) for v in obj["items"])
        if state_type == "dict":
            return {FsdpStrategy._json_restore(k): FsdpStrategy._json_restore(v) for k, v in obj["items"]}
        return {k: FsdpStrategy._json_restore(v) for k, v in obj.items()}

    @staticmethod
    def _atomic_write_json(path: str, payload) -> None:
        import json

        tmp_path = f"{path}.tmp.{os.getpid()}"
        with open(tmp_path, "w") as f:
            json.dump(FsdpStrategy._json_safe(payload), f)
        os.replace(tmp_path, path)

    def _write_ckpt_metric(self, ckpt_dir: str, metric_value, metric_key=None) -> None:
        import json

        path = self._get_ckpt_metric_path(ckpt_dir)
        tmp_path = f"{path}.tmp.{os.getpid()}"
        with open(tmp_path, "w") as f:
            json.dump({"metric_key": metric_key, "metric_value": metric_value}, f, indent=2, sort_keys=True)
        os.replace(tmp_path, path)

    def _read_ckpt_metric(self, ckpt_dir: str) -> float | None:
        import json

        metric_path = self._get_ckpt_metric_path(ckpt_dir)
        if not os.path.exists(metric_path):
            return None
        try:
            with open(metric_path) as f:
                payload = json.load(f)
            value = payload.get("metric_value") if isinstance(payload, dict) else None
            return None if value is None else float(value)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.print(f"Warning: failed to read checkpoint metric from {metric_path}: {exc}")
            return None

    @staticmethod
    def _dir_size(path: str) -> int:
        total = 0
        for dirpath, _, filenames in os.walk(path):
            for filename in filenames:
                try:
                    total += os.path.getsize(os.path.join(dirpath, filename))
                except OSError:
                    pass
        return total

    @staticmethod
    def _is_loadable_ckpt_dir(path: str) -> bool:
        return os.path.isdir(path) and os.path.isdir(os.path.join(path, "model"))

    def _checkpoint_candidates(self, ckpt_path: str, *, include_best: bool) -> list[tuple[str, float]]:
        if not os.path.isdir(ckpt_path):
            return []
        candidates = []
        for name in os.listdir(ckpt_path):
            path = os.path.join(ckpt_path, name)
            if not os.path.isdir(path):
                continue
            if not include_best and name.startswith("best"):
                continue
            if self._is_loadable_ckpt_dir(path):
                candidates.append((path, os.path.getmtime(path)))
        return candidates

    def _resolve_ckpt_load_dir(self, ckpt_path: str) -> str | None:
        latest = os.path.join(ckpt_path, "latest")
        if os.path.isfile(latest):
            with open(latest) as f:
                tag = f.read().strip()
            load_dir = os.path.join(ckpt_path, tag)
            if self._is_loadable_ckpt_dir(load_dir):
                if not tag.startswith("best"):
                    return load_dir
                regular = self._checkpoint_candidates(ckpt_path, include_best=False)
                if regular:
                    fallback = max(regular, key=lambda item: item[1])[0]
                    self.print(
                        f"Warning: latest points to best checkpoint {tag}; "
                        f"resuming from newest regular checkpoint {os.path.basename(fallback)}."
                    )
                    return fallback
                return load_dir
            self.print(f"Warning: latest checkpoint {load_dir} is missing or incomplete; scanning checkpoints.")

        regular = self._checkpoint_candidates(ckpt_path, include_best=False)
        if regular:
            fallback = max(regular, key=lambda item: item[1])[0]
            self.print(f"Warning: latest file missing/stale; resuming from {os.path.basename(fallback)}.")
            return fallback

        best = self._checkpoint_candidates(ckpt_path, include_best=True)
        best = [(path, mtime) for path, mtime in best if os.path.basename(path).startswith("best")]
        if best:
            fallback = max(best, key=lambda item: item[1])[0]
            self.print(f"Warning: only best checkpoints found; resuming from {os.path.basename(fallback)}.")
            return fallback
        return None

    def _prune_checkpoints(self, ckpt_path: str, current_tag: str, max_num: int, max_mem: int, is_best: bool) -> None:
        import shutil

        if is_best:
            for name in os.listdir(ckpt_path):
                path = os.path.join(ckpt_path, name)
                if name.startswith("best") and name != current_tag and os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
            return

        max_size_bytes = None
        if max_mem is not None and max_mem > 0 and not math.isinf(max_mem):
            max_size_bytes = max_mem * 1024**3

        while True:
            subdirs = [
                (os.path.join(ckpt_path, name), os.path.getmtime(os.path.join(ckpt_path, name)))
                for name in os.listdir(ckpt_path)
                if os.path.isdir(os.path.join(ckpt_path, name))
            ]
            regular_subdirs = [
                (path, mtime)
                for path, mtime in subdirs
                if not os.path.basename(path).startswith("best") and os.path.basename(path) != current_tag
            ]
            current_regular_count = sum(1 for path, _ in subdirs if not os.path.basename(path).startswith("best"))
            overflow_num = max(0, current_regular_count - max_num) if max_num and max_num > 0 else 0
            overflow_mem = (
                max_size_bytes is not None and sum(self._dir_size(path) for path, _ in subdirs) > max_size_bytes
            )
            if overflow_num == 0 and not overflow_mem:
                break
            candidates = sorted(
                [(path, self._read_ckpt_metric(path), mtime) for path, mtime in regular_subdirs],
                key=lambda item: (
                    item[1] is not None,
                    item[1] if item[1] is not None else float("-inf"),
                    item[2],
                ),
            )
            if not candidates:
                break
            shutil.rmtree(candidates[0][0], ignore_errors=True)

    def save_ckpt(
        self,
        model: nn.Module,
        ckpt_path: str,
        tag: str,
        max_num: int = 3,
        max_mem: int = 0,
        client_states=None,
        **kwargs,
    ) -> None:
        """DCP-format checkpoint for resumable training (model + optimizer +
        scheduler + RL stats). HF-safetensors export goes through ``save_model``.
        """
        peft_config = getattr(model, "peft_config", None)
        model = self._unwrap_model(model)
        is_rank0 = (not dist.is_initialized()) or dist.get_rank() == 0
        is_best = tag.startswith("best")

        if is_rank0:
            os.makedirs(ckpt_path, exist_ok=True)

        if dist.is_initialized():
            dist.barrier()

        save_dir = os.path.join(ckpt_path, tag)
        os.makedirs(save_dir, exist_ok=True)

        ckpt = self._build_checkpointer(save_dir, save_consolidated=False, model=model)
        ckpt.save_model(model=model, weights_path=save_dir, tokenizer=None, peft_config=peft_config)
        optimizer = kwargs.get("optimizer")
        scheduler = kwargs.get("scheduler")
        if optimizer is not None:
            ckpt.save_optimizer(optimizer=optimizer, model=model, weights_path=save_dir, scheduler=scheduler)

        if dist.is_initialized():
            dist.barrier()

        if is_rank0:
            extra = {"client_state": dict(client_states or {})}
            cfg = getattr(model, "config", None)
            if cfg is not None and getattr(cfg, "normalize_reward", False):
                extra["runtime_state"] = {
                    "normalize_reward": True,
                    "mean": float(getattr(cfg, "mean", 0.0)),
                    "std": float(getattr(cfg, "std", 1.0)),
                }
            self._atomic_write_json(os.path.join(save_dir, "extra_state.json"), extra)
            self._write_ckpt_metric(save_dir, kwargs.get("metric_value"), kwargs.get("metric_key"))
            if not is_best:
                self._atomic_write_text(os.path.join(ckpt_path, "latest"), tag)
            self._prune_checkpoints(ckpt_path, tag, max_num, max_mem, is_best)
        if dist.is_initialized():
            dist.barrier()

    def load_ckpt(self, model: nn.Module, ckpt_path: str, optimizer=None, scheduler=None, **kwargs):
        """Load the most recent DCP checkpoint under ``ckpt_path``. Returns
        ``(load_path, states)`` where ``states`` carries ``client_state`` keys
        (e.g. ``consumed_samples``); ``(None, {})`` if no checkpoint is found.
        """
        import json

        if not os.path.isdir(ckpt_path):
            return None, {}
        load_dir = self._resolve_ckpt_load_dir(ckpt_path)
        if load_dir is None:
            return None, {}

        wrapper = model
        model = self._unwrap_model(model)

        # Tied-weight load: AutoModel saves in HF safetensors shards with
        # .hf_metadata sidecar (not DCP .metadata), and tie_word_embeddings=True
        # models (e.g. Qwen2.5-0.5B) deduplicate the tied pair. Use HF storage
        # reader + allow_partial_load to handle both.
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint import HuggingFaceStorageReader
        from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_model_state_dict,
            set_model_state_dict,
        )

        model_state = get_model_state_dict(model, options=StateDictOptions(full_state_dict=False, cpu_offload=False))
        dcp.load(
            model_state,
            storage_reader=HuggingFaceStorageReader(path=os.path.join(load_dir, "model")),
            planner=DefaultLoadPlanner(allow_partial_load=True),
        )
        set_model_state_dict(model, model_state, options=StateDictOptions(strict=False))
        optim_dir = os.path.join(load_dir, "optim")
        if optimizer is not None and os.path.isdir(optim_dir):
            from nemo_automodel.components.checkpoint.stateful_wrappers import OptimizerState

            optimizer_state = OptimizerState(model, optimizer, scheduler, is_peft=self.is_peft)
            optim_state_dict = optimizer_state.state_dict()
            dcp.load(
                optim_state_dict,
                checkpoint_id=optim_dir,
                planner=DefaultLoadPlanner(allow_partial_load=True),
            )
            optimizer_state.load_state_dict(optim_state_dict)

        states = {}
        extra_path = os.path.join(load_dir, "extra_state.json")
        if os.path.isfile(extra_path):
            with open(extra_path) as f:
                extra = self._json_restore(json.load(f))
            states = extra.get("client_state", {}) or {}
            runtime = extra.get("runtime_state") or {}
            unwrapped = self._unwrap_model(model)
            cfg = getattr(unwrapped, "config", None)
            if runtime and cfg is not None:
                for k in ("normalize_reward", "mean", "std"):
                    if k in runtime:
                        setattr(cfg, k, runtime[k])
                if hasattr(wrapper, "mean") and "mean" in runtime:
                    wrapper.mean[0] = runtime["mean"]
                if hasattr(wrapper, "std") and "std" in runtime:
                    wrapper.std[0] = runtime["std"]
                if hasattr(wrapper, "normalize_reward") and "normalize_reward" in runtime:
                    wrapper.normalize_reward = runtime["normalize_reward"]
        if dist.is_initialized():
            dist.barrier()
        return load_dir, states

    def load_model(self, model: nn.Module, model_path: str, **kwargs) -> None:
        """Load HF-safetensors weights into an already-parallelized model.
        Initial weight load is normally handled by ``NeMoAutoModelForCausalLM.
        from_pretrained``; use this only for explicit later reloads.
        """
        model = self._unwrap_model(model)
        ckpt = self._build_checkpointer(model_path, save_consolidated=False)
        ckpt.load_model(model=model, model_path=model_path, use_checkpoint_id=False)
        if dist.is_initialized():
            dist.barrier()

    @torch.no_grad()
    def moving_average(self, model: nn.Module, ema_model: nn.Module, beta: float, device: str = "cuda") -> None:
        """In-place EMA update: ``ema = beta * ema + (1 - beta) * model``.
        Iterates over (param, ema_param) pairs by name; both sides are FSDP2-
        sharded DTensors so the lerp is local to each rank's shard.
        """
        src = self._unwrap_model(model)
        tgt = self._unwrap_model(ema_model)
        ema_params = dict(tgt.named_parameters())
        for name, param in src.named_parameters():
            ema_param = ema_params.get(name)
            if ema_param is None:
                continue
            ema_param.data.mul_(beta).add_(param.data, alpha=1.0 - beta)


def _torch_dtype(s: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[s]
