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
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers.trainer import get_scheduler

from openrlhf.utils.distributed_sampler import DistributedSampler


def _get_actor_cls():
    """Lazy import to avoid circular dep: openrlhf.models.actor imports from this package."""
    from openrlhf.models import Actor

    return Actor


class FsdpStrategy:
    """FSDP2 + TP/CP/SP/EP backend, wrapping NVIDIA-NeMo/Automodel.

    Mirrors DeepspeedStrategy's public surface so trainers are agnostic to the
    backend. The model is built and parallelized via Automodel's official
    entry point ``NeMoAutoModelForCausalLM.from_pretrained`` inside ``Actor``;
    this strategy only handles distributed setup, optimizer/scheduler
    construction, the train-step (loss backward, grad clip, optimizer step),
    collectives, and checkpointing. Grad norm / clip are imported directly
    from ``nemo_automodel.components.distributed.grad_utils``.
    """

    CKPT_METRIC_FILENAME = "metric.json"

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
        # Activation checkpointing — train_sft/rm/dpo expose it as
        # --model.gradient_checkpointing_enable; train_ppo_ray uses
        # --actor.gradient_checkpointing_enable (PPO has separate actor/critic).
        # Read from whichever namespace is present.
        _model_ns = getattr(args, "model", None)
        _actor_ns = getattr(args, "actor", None)
        self.activation_checkpointing = (
            getattr(_model_ns, "gradient_checkpointing_enable", None)
            if _model_ns is not None and hasattr(_model_ns, "gradient_checkpointing_enable")
            else getattr(_actor_ns, "gradient_checkpointing_enable", False)
        )
        self.cpu_offload = getattr(fsdp, "cpu_offload", False)
        sp = getattr(fsdp, "sequence_parallel", None)
        # SP defaults to ON whenever TP>1 (user-stated default).
        self.sequence_parallel = sp if sp is not None else (self.tp_size > 1)
        self.optim = getattr(args, "optim", "adam")
        self.use_dynamic_batch = getattr(args.train, "dynamic_batch_enable", False)

        self.world_size: int = 1
        self.device_mesh = None
        self.moe_mesh = None
        self.dp_cp_group = None
        self.tp_group = None
        self.dp_group = None
        self.accumulated_gradient: int = 1
        self._last_grad_norm: float = 0.0
        self.time_steps = defaultdict(int)

    # ProcessGroup / DeviceMesh aren't picklable. `datasets.map(self.process_data,
    # num_proc>1)` indirectly pickles the strategy via the dataset's bound method;
    # drop the distributed handles so the workers spawn cleanly. Workers don't need
    # them — they only run pure-CPU data preprocessing.
    _UNPICKLABLE_ATTRS = ("tp_group", "dp_cp_group", "dp_group", "device_mesh", "moe_mesh", "distributed_config")

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k not in self._UNPICKLABLE_ATTRS}

    def __setstate__(self, state):
        self.__dict__.update(state)
        for k in self._UNPICKLABLE_ATTRS:
            self.__dict__.setdefault(k, None)

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
            dist.init_process_group(backend="nccl", timeout=timeout)

        self.world_size = dist.get_world_size()

        from nemo_automodel.components.distributed.config import FSDP2Config
        from nemo_automodel.components.distributed.mesh_utils import create_device_mesh

        # Monkey-patch Automodel's TPLinear forward: their guard misses some
        # non-contiguous DTensor inputs and falls through to F.linear, which
        # then dies on aten.view (RuntimeError: view size is not compatible).
        # Always contiguify before F.linear; cheap when already contiguous.
        if self.tp_size > 1:
            from nemo_automodel.components.distributed import parallel_styles

            _TPLinear = parallel_styles.TPLinear
            if not getattr(_TPLinear, "_orlhf_patched", False):
                import torch.nn.functional as F
                from torch.distributed.tensor import DTensor

                def _safe_forward(self, x):
                    # Always avoid F.linear for DTensor inputs: F.linear's
                    # internal view fails on sharded DTensor regardless of
                    # is_contiguous(). Use mm/bmm directly which dispatch
                    # through DTensor's matmul strategy.
                    if isinstance(x, DTensor):
                        if x.dim() == 3:
                            b = x.shape[0]
                            out = torch.bmm(x, self.weight.t().unsqueeze(0).expand(b, -1, -1))
                        else:
                            out = torch.mm(x, self.weight.t())
                        return out + self.bias if self.bias is not None else out
                    return F.linear(x, self.weight, self.bias)

                _TPLinear.forward = _safe_forward
                _TPLinear._orlhf_patched = True

        torch_dtype = _torch_dtype(self.param_dtype)
        # FSDP2 mixed precision: params/forward in bf16, gradient reduce-scatter
        # in fp32. Reducing in bf16 accumulates rounding error across DP ranks
        # and inflates grad_norm (observed 60–130 vs. ~1 expected). cast_forward
        # _inputs handles the lm_head dtype mismatch by casting fp32 inputs to
        # bf16 at FSDP unit boundaries. Mirrors NeMo-Automodel's recipe.
        mp_policy = (
            None
            if self.param_dtype == "fp32"
            else MixedPrecisionPolicy(
                param_dtype=torch_dtype,
                reduce_dtype=torch.float32,
                cast_forward_inputs=True,
            )
        )
        # Public attribute — `Actor` reads it as `strategy.distributed_config`
        # and forwards to `NeMoAutoModelForCausalLM.from_pretrained`.
        self.distributed_config = FSDP2Config(
            sequence_parallel=self.sequence_parallel,
            activation_checkpointing=self.activation_checkpointing,
            mp_policy=mp_policy,
            offload_policy=CPUOffloadPolicy(pin_memory=False) if self.cpu_offload else None,
            # Force grad sync inside backward(): we read p.grad immediately after
            # backward() to compute the global grad_norm. With deferred sync,
            # each rank sees its un-reduced local grad and the global norm
            # comes out √dp_size× too high (observed: SFT step1 23 vs PR#1176
            # reference 13 with dp=3, ratio √3 ≈ 1.73).
            defer_fsdp_grad_sync=False,
        )

        self.device_mesh, self.moe_mesh = create_device_mesh(
            self.distributed_config,
            tp_size=self.tp_size,
            pp_size=self.pp_size,
            cp_size=self.cp_size,
            ep_size=self.ep_size,
            world_size=self.world_size,
        )

        # Process groups for grad-norm/clip and data loaders.
        # Mesh dim names follow Automodel's FSDP2 convention.
        if self.tp_size > 1 and "tp" in self.device_mesh.mesh_dim_names:
            self.tp_group = self.device_mesh["tp"].get_group()

        # dp_shard_cp = data-parallel shard × context-parallel; this is the group
        # over which Automodel's grad_utils reduces.
        if "dp_shard_cp" in self.device_mesh.mesh_dim_names:
            self.dp_cp_group = self.device_mesh["dp_shard_cp"].get_group()
        elif "dp" in self.device_mesh.mesh_dim_names:
            self.dp_cp_group = self.device_mesh["dp"].get_group()

        if "dp_shard" in self.device_mesh.mesh_dim_names:
            self.dp_group = self.device_mesh["dp_shard"].get_group()
        elif "dp" in self.device_mesh.mesh_dim_names:
            self.dp_group = self.device_mesh["dp"].get_group()

        dp_size = dist.get_world_size(group=self.dp_group) if self.dp_group else self.world_size
        # Effective grad-accum = train_batch_size / (micro_bs × DP).
        self.accumulated_gradient = max(self.train_batch_size // (self.micro_train_batch_size * dp_size), 1)

    # ---------------------------------------------------------------- prepare

    def prepare(self, *args):
        ret = []
        for arg in args:
            if isinstance(arg, tuple):
                assert len(arg) == 2, f"prepare() tuple must be (model, cfg); got len={len(arg)}"
                model, cfg = arg
                if model is None:
                    ret.append((None, None, None))
                else:
                    ret.append(self._init_train_model(model, cfg))
            else:
                ret.append(self._init_eval_model(arg))
        return ret[0] if len(ret) == 1 else ret

    def _init_train_model(self, model, cfg: dict):
        # Model is already parallelized — Actor builds via
        # NeMoAutoModelForCausalLM.from_pretrained (Automodel's official entry),
        # which handles FSDP2 wrap + TP plan + CP hooks internally given the
        # device_mesh + distributed_config we expose on this strategy.
        is_actor = isinstance(model, _get_actor_cls())
        params = [p for p in (model.model if is_actor else model).parameters() if p.requires_grad]

        kind = cfg.get("optim", self.optim)
        if kind == "muon":
            raise NotImplementedError("Muon under FSDP2 not yet wired; use --optim adam")

        adam = cfg["adam"]
        optimizer = torch.optim.AdamW(
            params,
            lr=adam["lr"],
            betas=tuple(adam["betas"]),
            eps=adam["eps"],
            weight_decay=adam["weight_decay"],
            foreach=False,
            fused=False,
        )

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
        # Eval models are also built+parallelized via Automodel's official entry
        # at construction time (Actor or get_llm_for_sequence_regression).
        return model

    # ---------------------------------------------------------------- step loop

    def backward(
        self,
        loss: torch.Tensor,
        model: nn.Module,
        optimizer: optim.Optimizer,
        name: str = "model",
        **kwargs,
    ) -> None:
        # FSDP2's reduce-scatter averages grads across the DP shard group, so
        # per-token-mean losses reduce to a correct cross-rank mean directly.
        # Do NOT multiply the loss by dp_size — that triple-counts and inflates
        # grad_norm (observed: SFT step1 grad ~69 with DP=3 → ~23 after removal).
        if self.accumulated_gradient > 1:
            loss = loss / self.accumulated_gradient
            # Skip the DP all-reduce on intermediate accum steps (saves bandwidth);
            # only sync on the final micro-batch where optimizer_step will fire.
            # The key MUST match optimizer_step's so the counter advances together.
            unwrapped = self._unwrap_model(model)
            if hasattr(unwrapped, "set_requires_gradient_sync"):
                key = f"step_{name}"
                sync = (self.time_steps[key] + 1) % self.accumulated_gradient == 0
                unwrapped.set_requires_gradient_sync(sync)
        loss.backward()

    def optimizer_step(
        self,
        optimizer: optim.Optimizer,
        model: nn.Module,
        scheduler,
        name: str = "model",
        **kwargs,
    ) -> None:
        # Gradient accumulation: skip the optimizer step until the last micro
        # batch in the accumulation window has run. Mirrors PR#1176 / NeMo-RL.
        key = f"step_{name}"
        self.time_steps[key] += 1
        if self.time_steps[key] % self.accumulated_gradient != 0:
            return

        # DTensor-aware grad norm + clip via torch.nn.utils. Automodel's
        # get_grad_norm/clip_grad_by_total_norm_ over-counts on FSDP-sharded
        # grads with cp=1: each rank's local shard already represents the
        # post-reduce-scatter averaged grad, but the function then all-reduces
        # SUM across dp_cp_group again — inflating norm by √dp_size (observed:
        # 23 vs PR#1176 reference 13 with dp=3, ratio √3 ≈ 1.73).
        if isinstance(model, _get_actor_cls()):
            model = model.model
        params = [p for p in model.parameters() if p.grad is not None]
        if not params:
            self._last_grad_norm = 0.0
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            return
        # Group by (mesh, placements): different DTensor sharding patterns
        # cannot be flattened into one norm call. PR#1176 / NeMo-RL use the
        # same grouping idea; mostly one group at TP=1.
        from torch.distributed.tensor import DTensor

        groups: dict = {}
        for p in params:
            g = p.grad
            key = (g.device_mesh, g.placements) if isinstance(g, DTensor) else ("plain",)
            groups.setdefault(key, []).append(p)

        total_norm_sq = torch.zeros((), dtype=torch.float32, device=torch.cuda.current_device())
        for group in groups.values():
            grads = [p.grad for p in group]
            partial = torch.nn.utils.get_total_norm(grads, norm_type=2.0, error_if_nonfinite=False)
            if isinstance(partial, DTensor):
                partial = partial.full_tensor()
            total_norm_sq = total_norm_sq + partial.to(total_norm_sq.dtype) ** 2
        total_norm = total_norm_sq.sqrt()
        self._last_grad_norm = float(total_norm.item())
        if self.max_norm > 0:
            for group in groups.values():
                torch.nn.utils.clip_grads_with_norm_(group, max_norm=self.max_norm, total_norm=total_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    def get_grad_norm(self, model: nn.Module) -> float:
        return self._last_grad_norm

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
        if sampler is None and dist.is_initialized() and self.dp_group is not None:
            num_replicas = dist.get_world_size(group=self.dp_group)
            rank = dist.get_rank(group=self.dp_group)
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

    # ---------------------------------------------------------------- compat shims

    def _unwrap_model(self, model) -> nn.Module:
        if isinstance(model, _get_actor_cls()):
            return self._unwrap_model(model.model)
        if hasattr(model, "module"):
            return model.module
        return model

    def get_ds_train_config(self, *args, **kwargs):
        # Returned dict is consumed by Actor for HfDeepSpeedConfig under the DS
        # backend; under fsdp it's a no-op. Trainers pass it through unchanged.
        return None

    def get_ds_eval_config(self, *args, **kwargs):
        return None

    # ---------------------------------------------------------------- I/O (MVP)

    def save_model(self, model: nn.Module, tokenizer, output_dir: str, **kwargs) -> None:
        # Use Automodel's Checkpointer — its custom-model save_pretrained mixin
        # requires it (raises "No checkpointer provided" otherwise). Outputs
        # consolidated HF safetensors that vLLM can hot-load.
        from nemo_automodel.components.checkpoint.checkpointing import Checkpointer, CheckpointingConfig

        if isinstance(model, _get_actor_cls()):
            model = model.model

        os.makedirs(output_dir, exist_ok=True)
        config = CheckpointingConfig(
            enabled=True,
            checkpoint_dir=output_dir,
            model_save_format="safetensors",
            model_cache_dir="",
            model_repo_id="",
            save_consolidated=True,
            is_peft=False,
            v4_compatible=True,  # produce HF-style config.json that vLLM consumes
        )
        ckpt = Checkpointer(
            config=config,
            dp_rank=dist.get_rank(group=self.dp_group) if self.dp_group else 0,
            tp_rank=dist.get_rank(group=self.tp_group) if self.tp_group else 0,
            pp_rank=0,
            moe_mesh=self.moe_mesh,
        )
        ckpt.save_model(model=model, weights_path=output_dir, tokenizer=tokenizer)
        dist.barrier()

    def _build_checkpointer(self, output_dir: str, save_consolidated: bool):
        from nemo_automodel.components.checkpoint.checkpointing import Checkpointer, CheckpointingConfig

        os.makedirs(output_dir, exist_ok=True)
        config = CheckpointingConfig(
            enabled=True,
            checkpoint_dir=output_dir,
            model_save_format="safetensors",
            model_cache_dir="",
            model_repo_id="",
            save_consolidated=save_consolidated,
            is_peft=False,
            v4_compatible=True,
        )
        return Checkpointer(
            config=config,
            dp_rank=dist.get_rank(group=self.dp_group) if self.dp_group else 0,
            tp_rank=dist.get_rank(group=self.tp_group) if self.tp_group else 0,
            pp_rank=0,
            moe_mesh=self.moe_mesh,
        )

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
        import json
        import shutil

        if isinstance(model, _get_actor_cls()):
            model = model.model

        save_dir = os.path.join(ckpt_path, tag)
        os.makedirs(save_dir, exist_ok=True)
        is_rank0 = (not dist.is_initialized()) or dist.get_rank() == 0

        ckpt = self._build_checkpointer(save_dir, save_consolidated=False)
        ckpt.save_model(model=model, weights_path=save_dir, tokenizer=None)

        if is_rank0:
            extra = {"client_state": dict(client_states or {})}
            unwrapped = self._unwrap_model(model)
            cfg = getattr(unwrapped, "config", None)
            if cfg is not None and getattr(cfg, "normalize_reward", False):
                extra["runtime_state"] = {
                    "normalize_reward": True,
                    "mean": float(getattr(cfg, "mean", 0.0)),
                    "std": float(getattr(cfg, "std", 1.0)),
                }
            with open(os.path.join(save_dir, "extra_state.json"), "w") as f:
                json.dump(extra, f)
            with open(os.path.join(ckpt_path, "latest"), "w") as f:
                f.write(tag)
            if max_num and max_num > 0:
                tags = sorted(
                    (d for d in os.listdir(ckpt_path) if os.path.isdir(os.path.join(ckpt_path, d))),
                    key=lambda d: os.path.getmtime(os.path.join(ckpt_path, d)),
                )
                for old in tags[:-max_num]:
                    shutil.rmtree(os.path.join(ckpt_path, old), ignore_errors=True)
        if dist.is_initialized():
            dist.barrier()

    def load_ckpt(self, model: nn.Module, ckpt_path: str, **kwargs):
        """Load the most recent DCP checkpoint under ``ckpt_path``. Returns
        ``(load_path, states)`` where ``states`` carries ``client_state`` keys
        (e.g. ``consumed_samples``); ``(None, {})`` if no checkpoint is found.
        """
        import json

        if not os.path.isdir(ckpt_path):
            return None, {}
        latest = os.path.join(ckpt_path, "latest")
        if os.path.isfile(latest):
            with open(latest) as f:
                tag = f.read().strip()
            load_dir = os.path.join(ckpt_path, tag)
        else:
            return None, {}
        if not os.path.isdir(load_dir):
            return None, {}

        if isinstance(model, _get_actor_cls()):
            model = model.model

        ckpt = self._build_checkpointer(load_dir, save_consolidated=False)
        ckpt.load_model(model=model, model_path=load_dir, use_checkpoint_id=False)

        states = {}
        extra_path = os.path.join(load_dir, "extra_state.json")
        if os.path.isfile(extra_path):
            with open(extra_path) as f:
                extra = json.load(f)
            states = extra.get("client_state", {}) or {}
            runtime = extra.get("runtime_state") or {}
            unwrapped = self._unwrap_model(model)
            cfg = getattr(unwrapped, "config", None)
            if runtime and cfg is not None:
                for k in ("normalize_reward", "mean", "std"):
                    if k in runtime:
                        setattr(cfg, k, runtime[k])
        if dist.is_initialized():
            dist.barrier()
        return load_dir, states

    def load_model(self, model: nn.Module, model_path: str, **kwargs) -> None:
        """Load HF-safetensors weights into an already-parallelized model.
        Initial weight load is normally handled by ``NeMoAutoModelForCausalLM.
        from_pretrained``; use this only for explicit later reloads.
        """
        if isinstance(model, _get_actor_cls()):
            model = model.model
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
