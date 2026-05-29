import gc
import math
import os
from abc import ABC
from collections import defaultdict
from datetime import timedelta
from functools import partial
from typing import Optional

import deepspeed
import torch
import torch.nn as nn
import torch.optim as optim
import transformers
import transformers.modeling_flash_attention_utils
from torch import distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from transformers.trainer import get_scheduler

from openrlhf.models import Actor
from openrlhf.models.ring_attn_utils import get_ring_attn_group, set_ring_attn_group

from .deepspeed_utils import (
    _z3_params_to_fetch,
    get_eval_ds_config,
    get_optimizer_grouped_parameters,
    get_train_ds_config,
)


class DeepspeedStrategy(ABC):
    """
    The strategy for training with Accelerator.
    """

    def __init__(
        self,
        seed: int = 42,
        full_determinism: bool = False,
        max_norm: float = 0.0,
        micro_train_batch_size=1,
        train_batch_size=1,
        zero_stage=2,
        args=None,
    ) -> None:
        super().__init__()

        self.args = args
        self.stage = zero_stage
        self.train_batch_size = train_batch_size
        self.micro_train_batch_size = micro_train_batch_size
        self.param_dtype = args.ds.param_dtype  # default: bf16
        self.seed = seed
        self.full_determinism = full_determinism
        self.max_norm = max_norm

        self.optim = getattr(args, "optim", "adam")
        self.adam_offload = getattr(args.ds, "adam_offload", False)
        self.zpg = getattr(args.ds, "zpg", 1)
        self.use_ds_universal_ckpt = getattr(args.ds, "use_universal_ckpt", False)
        self.grad_accum_dtype = getattr(args.ds, "grad_accum_dtype", None)
        self.overlap_comm = getattr(args.ds, "overlap_comm", False)
        self.deepcompile = getattr(args.ds, "deepcompile", False)
        self.ds_tensor_parallel_size = getattr(args.ds, "tensor_parallel_size", 1)
        self.ring_attn_size = getattr(args.ds, "ring_attn_size", 1)
        self.use_dynamic_batch = getattr(self.args.train, "dynamic_batch_enable", False)

        if self.ds_tensor_parallel_size > 1:
            assert deepspeed.version >= "0.16.4", "DeepSpeed version must be >= 0.16.4 for tensor parallel training"
            assert self.param_dtype == "bf16", "BF16 is required for tensor parallel training"

        self.time_steps = defaultdict(int)

    def setup_distributed(self, timeout=timedelta(minutes=60)) -> None:
        if self.full_determinism:
            transformers.enable_full_determinism(self.seed)
            # Use deterministic backward in flash attention as, by default, flash attention uses atomic adds
            # https://github.com/Dao-AILab/flash-attention/commit/732654583c2e640adc012ecb60e460bf19dcd9e3
            transformers.modeling_flash_attention_utils.deterministic_g = True
        else:
            transformers.set_seed(self.seed)

        # Take the local rank from args as first priority
        if self.args.local_rank != -1:
            os.environ["LOCAL_RANK"] = str(self.args.local_rank)

        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        if local_rank != -1:
            torch.cuda.set_device(local_rank)

        # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
        deepspeed.init_distributed(timeout=timeout)

        # mesh
        self.world_size = dist.get_world_size()
        dp_size = self.world_size // self.ring_attn_size // self.ds_tensor_parallel_size
        self.ds_device_mesh = init_device_mesh(
            "cuda", (dp_size, self.ring_attn_size, self.ds_tensor_parallel_size), mesh_dim_names=("dp", "sp", "tp")
        )
        self.setup_ring_attn(self.ds_device_mesh)

        self.accumulated_gradient = (
            self.train_batch_size
            * self.ring_attn_size
            * self.ds_tensor_parallel_size
            // self.micro_train_batch_size
            // self.world_size
        )

    def setup_ring_attn(self, ds_device_mesh):
        if self.ring_attn_size == 1:
            self.ring_attn_rank = 0
            return

        # get the group of the current device
        group = ds_device_mesh["sp"].get_group()
        self.ring_attn_rank = dist.get_rank(group=group)
        set_ring_attn_group(group)

        from openrlhf.models.ring_attn_utils import patch_transformers_for_ring_flash_attn

        patch_transformers_for_ring_flash_attn()
        from ring_flash_attn import substitute_hf_flash_attn

        self.ring_head_stride = getattr(self.args.ds, "ring_attn_head_stride", 1)
        substitute_hf_flash_attn(self.ring_attn_group, self.ring_head_stride)

    @property
    def ring_attn_group(self):
        return get_ring_attn_group()

    def _get_model_parameters(self, model: nn.Module, *, optim: Optional[str] = None, weight_decay: float = 0.0):
        # ``optim`` defaults to the strategy-wide setting but can be overridden per-model
        # so actor/critic can disagree (e.g. actor=muon, critic=adam).
        # ``weight_decay`` is the user-requested decay for the decaying group;
        # bias/LayerNorm are always exempted (weight_decay=0.0) for the Adam path.
        if optim is None:
            optim = self.optim
        raw_model = model.model if isinstance(model, Actor) else model
        if optim == "muon":
            from packaging import version

            # 0.18.9 is the first release that honors `muon_lr` / `adam_lr` config
            # overrides (engine.py:1757-1771). Earlier 0.18.x silently drops them,
            # causing aux-Adam to inherit the Muon lr — ~20000x the intended value.
            assert version.parse(deepspeed.__version__) >= version.parse(
                "0.18.9"
            ), f"Muon optimizer requires deepspeed >= 0.18.9, got {deepspeed.__version__}"
            # DS sets `use_muon` itself inside `deepspeed.initialize` via
            # set_optimizer_flags (__init__.py:71-77). We pass raw grad-enabled params
            # and let DS handle partitioning. DS applies one scalar weight_decay to
            # both the Muon and aux-Adam groups, so bias/LayerNorm decay exemption
            # is NOT applied under Muon — warn when the user sets weight_decay>0.
            if weight_decay > 0:
                import warnings as _warnings

                _warnings.warn(
                    f"Muon + weight_decay={weight_decay} will also decay bias/LayerNorm "
                    f"(the Adam path exempts them). DS applies one scalar weight_decay to "
                    f"both the Muon and aux-Adam groups.",
                    stacklevel=2,
                )
            model_parameters = [param for param in raw_model.parameters() if param.requires_grad]
        else:
            # Adam: exempt bias/layernorm from weight decay.
            model_parameters = get_optimizer_grouped_parameters(raw_model, weight_decay)
        return raw_model, model_parameters

    def backward(self, loss: torch.Tensor, model: nn.Module, _optimizer: optim.Optimizer, **kwargs) -> None:
        # _optimizer kept for strategy interface compatibility; DS engine handles it internally.
        if isinstance(model, Actor):
            model = model.model
        model.backward(loss)

    def _unwrap_model(self, model) -> nn.Module:
        if isinstance(model, Actor):
            return self._unwrap_model(model.model)
        elif hasattr(model, "module"):
            return model.module
        else:
            return model

    def prepare(self, *args):
        """Prepare models for training/evaluation.

        Each arg is one of:

        * ``(model, cfg)`` — training.  ``cfg`` is a dict like::

            {
              "optim": "muon" | "adam",          # selector
              "muon":  {lr, momentum},           # Muon-specific; lr shared w/ aux-Adam
              "adam":  {lr, betas, eps, weight_decay},
              "lr_scheduler": str,
              "lr_warmup_ratio": float,
              "min_lr_ratio": float,
              "max_norm": float,                 # gradient clip
              "scheduler_steps": int,
            }

          When ``optim="muon"``, the Muon group uses ``muon.lr`` / ``muon.momentum``;
          the aux-Adam subgroup uses ``adam.lr`` / ``adam.betas`` / ``adam.eps``;
          ``adam.weight_decay`` is shared (DS applies one scalar ``weight_decay`` to both groups).
        * ``model`` — evaluation only, no optimizer.

        Returns ``(model, optimizer, scheduler)`` for training entries and the
        wrapped model for eval entries.
        """
        ret = []
        for arg in args:
            if isinstance(arg, tuple):
                assert len(arg) == 2, f"prepare() tuple must be (model, cfg); got len={len(arg)}"
                model, cfg = arg
                if model is None:
                    ret.append((None, None, None))
                else:
                    ret.append(self._ds_init_train_model(model, cfg))
            else:
                ret.append(self._ds_init_eval_model(arg))

        return ret[0] if len(ret) == 1 else ret

    def _ds_init_train_model(self, model, cfg: dict):
        # DS v0.18.9 optimizer-config whitelists (engine.py:1755-1772):
        #   AdamW:  {lr, betas, eps, weight_decay}
        #   Muon:   muon group -> {lr, momentum, weight_decay, muon_lr}
        #           aux-Adam   -> {lr, betas, eps, weight_decay, adam_lr}
        # ``muon_lr`` / ``adam_lr`` override ``lr`` per-group.  Keys outside these
        # sets are silently dropped.  ns_steps / nesterov are NOT configurable via
        # DS config in 0.18.x (hard-coded: ns_steps=5, nesterov=True).
        kind = cfg["optim"]
        adam = cfg["adam"]
        if kind == "muon":
            muon = cfg["muon"]
            # DS 0.18.9 engine.py:1757-1771 reads `muon_lr` / `adam_lr` from the
            # config and uses them to OVERRIDE the per-group lr at param-group
            # construction. Emit both explicitly so the two groups follow their
            # own initial lrs; the HF scheduler then drives each group independently.
            # (Earlier 0.18.x silently dropped these keys, causing aux-Adam to
            # inherit the Muon lr — ~20000x the intended value. The version
            # assert in _get_model_parameters guards against that.)
            #
            # ns_steps / nesterov: DS v0.18.x Muon hard-codes ns_steps=5, nesterov=True
            # inside muon_update() and does NOT accept them via config. We expose the
            # CLI knobs as placeholders; warn when the user sets a non-default value.
            if muon.get("ns_steps", 5) != 5 or not muon.get("nesterov", True):
                import warnings as _warnings
                import deepspeed as _ds

                _warnings.warn(
                    f"muon.ns_steps / muon.nesterov are placeholders: DeepSpeed "
                    f"{getattr(_ds, '__version__', '?')} hard-codes ns_steps=5, "
                    f"nesterov=True inside muon_update() and ignores config overrides.",
                    stacklevel=2,
                )
            # DS applies clipping AFTER muon_update replaces the raw grad with
            # the Newton-Schulz output, so the reported `grad_norm` is the NS
            # output's Frobenius norm (~sqrt(sum min(m,n)) ≈ 700 for a 1.5B
            # model). An Adam-scale max_norm (e.g. 1.0) will then scale the NS
            # update down ~700x and effectively kill Muon. Warn loudly — users
            # should pass `max_norm=0` (disable) until DS fixes the ordering.
            if cfg.get("max_norm", 0) > 0:
                import warnings as _warnings

                _warnings.warn(
                    f"Muon + gradient_clipping={cfg['max_norm']}: DeepSpeed applies "
                    f"the clip AFTER Newton-Schulz, so any positive max_norm is "
                    f"measured against the post-NS scale (~sqrt(sum min(m,n))). "
                    f"For Muon runs, set `max_norm=0` to disable global clipping "
                    f"(NS already bounds per-matrix spectral norm to 1).",
                    stacklevel=2,
                )
            optim_dict = {
                "type": "Muon",
                "params": {
                    "lr": muon["lr"],  # fallback for param-groups with no override
                    "muon_lr": muon["lr"],  # explicit override for the Muon group
                    "adam_lr": adam["lr"],  # explicit override for the aux-Adam group
                    "momentum": muon["momentum"],
                    "betas": list(adam["betas"]),  # aux-Adam only
                    "eps": adam["eps"],  # aux-Adam only
                    "weight_decay": adam["weight_decay"],  # shared by both groups (DS scalar)
                },
            }
        else:
            optim_dict = {
                "type": "AdamW",
                "params": {
                    "lr": adam["lr"],
                    "betas": list(adam["betas"]),
                    "eps": adam["eps"],
                    "weight_decay": adam["weight_decay"],
                },
            }
        scheduler_steps = cfg["scheduler_steps"]
        sched_factory = partial(
            get_scheduler,
            cfg.get("lr_scheduler", "cosine_with_min_lr"),
            num_warmup_steps=math.ceil(scheduler_steps * cfg.get("lr_warmup_ratio", 0.03)),
            num_training_steps=scheduler_steps,
            scheduler_specific_kwargs={"min_lr_rate": cfg.get("min_lr_ratio", 0.1)},
        )

        is_actor = isinstance(model, Actor)
        ds_config = self.get_ds_train_config(optim_dict=optim_dict, max_norm=cfg.get("max_norm", 1.0))

        if self.ds_tensor_parallel_size > 1:
            tp_model = deepspeed.tp_model_init(
                model=model.model if is_actor else model, tp_size=self.ds_tensor_parallel_size, dtype=torch.bfloat16
            )
            if is_actor:
                model.model = tp_model
            else:
                model = tp_model
            gc.collect()
            torch.cuda.empty_cache()

        # Infer optim kind from the DS type so actor/critic can disagree.
        optim_kind = "muon" if optim_dict.get("type") == "Muon" else "adam"
        raw_model, model_parameters = self._get_model_parameters(
            model, optim=optim_kind, weight_decay=adam["weight_decay"]
        )

        # DS classifies params via set_optimizer_flags (__init__.py:71-77 on 0.18.9):
        # ndim>=2 and name not containing "embed"/"lm_head" → Muon, else aux-Adam.
        # This works for LLaMA/Mistral/Qwen actors. Edge cases (GPT-2 wte/wpe, RM
        # value heads, LoRA A/B matrices) are classified to Muon by the default rule
        # — acceptable for now; add a per-parameter override here if we need to
        # exempt a specific head / adapter weight.
        engine, optim, _, scheduler = deepspeed.initialize(
            model=raw_model,
            optimizer=None,
            model_parameters=model_parameters,
            lr_scheduler=sched_factory,
            config=ds_config,
            args={"local_rank": int(os.environ.get("LOCAL_RANK", "-1"))},
            dist_init_required=True,
        )

        if self.deepcompile:
            engine.compile()
        if is_actor:
            model.model = engine
        else:
            model = engine

        return model, optim, scheduler

    def get_ds_train_config(
        self, optim_dict: Optional[dict] = None, *, max_norm: Optional[float] = None, is_actor=None
    ):
        # ``optim_dict`` is None when called from HF model-loading paths (e.g.
        # ``Actor.from_pretrained``) that only need the ZeRO + bf16 parts — the
        # optimizer section is filled later at ``prepare()`` time.
        # ``max_norm`` is per-model (set by prepare's cfg); falls back to
        # ``self.max_norm`` if not provided (for legacy model-loading callers).
        # ``is_actor`` is accepted for legacy call sites and unused here.
        del is_actor
        ds_config = get_train_ds_config(
            offload=False,
            adam_offload=self.adam_offload,
            stage=self.stage,
            param_dtype=self.param_dtype,
            max_norm=max_norm if max_norm is not None else self.max_norm,
            zpg=self.zpg,
            grad_accum_dtype=self.grad_accum_dtype,
            overlap_comm=self.overlap_comm,
            use_ds_universal_ckpt=self.use_ds_universal_ckpt,
            deepcompile=self.deepcompile,
            tensor_parallel_size=self.ds_tensor_parallel_size,
            optim_config=optim_dict,
        )
        if self.use_dynamic_batch:
            ds_config["train_micro_batch_size_per_gpu"] = 1
            ds_config["gradient_accumulation_steps"] = 1
        else:
            ds_config["train_micro_batch_size_per_gpu"] = self.micro_train_batch_size
            ds_config["train_batch_size"] = self.train_batch_size * self.ring_attn_size * self.ds_tensor_parallel_size

        return ds_config

    def _ds_init_eval_model(self, model):
        if not model:
            return model
        is_actor = isinstance(model, Actor)
        ds_config = self.get_ds_eval_config(offload=getattr(model, "_offload", False))

        if self.ds_tensor_parallel_size > 1:
            tp_model = deepspeed.tp_model_init(
                model=model.model if is_actor else model, tp_size=self.ds_tensor_parallel_size, dtype=torch.bfloat16
            )
            if isinstance(model, Actor):
                model.model = tp_model
            else:
                model = tp_model

        engine, *_ = deepspeed.initialize(
            model=model.model if is_actor else model,
            args={"local_rank": int(os.environ.get("LOCAL_RANK", "-1"))},
            config=ds_config,
            dist_init_required=True,
        )
        if self.deepcompile:
            engine.compile()
        if is_actor:
            model.model = engine
        else:
            model = engine
        return model

    def get_ds_eval_config(self, offload=False):
        # DS Config
        ds_config = get_eval_ds_config(
            offload=offload,
            stage=self.stage if self.stage == 3 else 0,
            param_dtype=self.param_dtype,
            deepcompile=self.deepcompile,
            tensor_parallel_size=self.ds_tensor_parallel_size,
        )
        ds_config["train_micro_batch_size_per_gpu"] = self.micro_train_batch_size
        ds_config["train_batch_size"] = self.train_batch_size * self.ring_attn_size * self.ds_tensor_parallel_size

        return ds_config

    def all_reduce(self, data, op="mean"):
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
        if self.is_rank_0():
            print(*msg)

    def is_rank_0(self) -> bool:
        if not dist.is_initialized():
            return True
        return dist.get_rank() == 0

    def get_rank(self) -> int:
        if not dist.is_initialized():
            return 0
        return dist.get_rank()

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
        assert isinstance(model, deepspeed.DeepSpeedEngine)
        load_path, states = model.load_checkpoint(
            load_dir,
            tag,
            load_module_strict=load_module_strict,
            load_optimizer_states=load_optimizer_states,
            load_lr_scheduler_states=load_lr_scheduler_states,
            load_module_only=load_module_only,
        )
        if load_path is None:
            self.print(f"Warning: [deepspeed] No checkpoint found at {load_dir}, skipping checkpoint loading.")
            return None, {}
        return load_path, states
