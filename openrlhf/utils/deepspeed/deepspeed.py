import gc
import json
import math
import os
import shutil
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
from peft import PeftModel, get_peft_model_state_dict
from torch import distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers.trainer import get_scheduler

from openrlhf.models import Actor
from openrlhf.models.ring_attn_utils import get_ring_attn_group, set_ring_attn_group
from openrlhf.utils.distributed_sampler import DistributedSampler
from openrlhf.utils.distributed_util import torch_dist_barrier_and_cuda_sync

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

    CKPT_METRIC_FILENAME = "metric.json"

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

            assert version.parse(deepspeed.__version__) >= version.parse(
                "0.18.2"
            ), f"Muon optimizer requires deepspeed >= 0.18.2, got {deepspeed.__version__}"
            # Muon for internal 2-D weight matrices; Adam for embeddings, classifier/value heads, and 1-D params.
            # DS Muon dispatches based on the `use_muon` attribute per parameter.
            # Covers LLaMA/Mistral/Qwen (embed_tokens), GPT-2 (wte/wpe), and RM/PPO critic value heads.
            _muon_exclude = {"embed", "lm_head", "wte", "wpe"}
            value_head_prefix = getattr(raw_model, "value_head_prefix", None)
            for name, param in raw_model.named_parameters():
                top_level_name = name.split(".", 1)[0]
                is_value_head = value_head_prefix is not None and top_level_name == value_head_prefix
                param.use_muon = param.ndim >= 2 and not is_value_head and not any(k in name for k in _muon_exclude)
            # DS Muon (engine.py 1597-1611) partitions a flat Parameter list via p.use_muon
            # and refuses param-group dicts; it also applies a single scalar weight_decay
            # to both the Muon and aux-Adam groups. Splitting groups post-init would desync
            # ZeRO's cached bit16_groups/fp32_groups metadata (new groups never get stepped),
            # so bias/LayerNorm decay exemption under Muon is NOT applied here -- DS limitation.
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

    def optimizer_step(
        self,
        _optimizer: optim.Optimizer,
        model: nn.Module,
        _scheduler,
        name="model",
        **kwargs,
    ) -> None:
        # Parameters are kept for strategy interface compatibility.
        if isinstance(model, Actor):
            model = model.model
        model.step()

    def get_grad_norm(self, model: nn.Module) -> float:
        """Get the global gradient norm from DeepSpeed engine (pre-clipping)."""
        if isinstance(model, Actor):
            model = model.model
        if hasattr(model, "get_global_grad_norm"):
            grad_norm = model.get_global_grad_norm()
            if grad_norm is None:
                return 0.0
            return grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
        return 0.0

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
        num_workers: int = 0,
    ):
        # DDP only mode, replay buffers on each rank are different.
        if sampler is None and dist.is_initialized():
            dp_group = self.ds_device_mesh["dp"].get_group()
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
          the aux-Adam subgroup uses ``adam.betas`` / ``adam.eps``; ``adam.weight_decay``
          is shared (DS v0.18.2 uses one ``weight_decay`` for both groups).
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
        # DS v0.18.2 optimizer-config whitelists (engine.py:1600-1611):
        #   AdamW:  {lr, betas, eps, weight_decay}
        #   Muon:   muon group -> {lr, momentum, weight_decay}
        #           aux-Adam   -> {lr, betas, eps, weight_decay}   (lr/weight_decay shared)
        # Keys outside these sets are silently dropped.  ns_steps / nesterov are
        # NOT configurable via DS config in v0.18.2 (hard-coded: ns_steps=5, nesterov=True).
        kind = cfg["optim"]
        adam = cfg["adam"]
        if kind == "muon":
            muon = cfg["muon"]
            # DS engine.py:1645,1654 reads `muon_lr` / `adam_lr` from the config
            # and uses them to OVERRIDE the per-group lr at param-group construction.
            # Without these keys, both groups silently inherit the same top-level lr —
            # which meant embeddings/head/value_head were trained at Muon lr (0.02).
            # Emit both explicitly so the two groups follow their own initial lrs;
            # the HF scheduler then drives each group independently.
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

        # DS __init__.py:69-75 (set_optimizer_flags) overwrites every param's
        # use_muon attribute using a substring rule that mis-classifies value_head
        # (sent to Muon — wrong per Keller Jordan) and GPT-2 wte/wpe (also sent to
        # Muon because they do not contain the "embed" substring). Our
        # _get_model_parameters() above already set use_muon correctly with the
        # project-specific exclusions; no-op DS's override for the duration of
        # initialize() and restore afterwards.
        _ds_set_flags_orig = getattr(deepspeed, "set_optimizer_flags", None)
        if optim_kind == "muon" and _ds_set_flags_orig is not None:
            deepspeed.set_optimizer_flags = lambda *args, **kwargs: None
        try:
            engine, optim, _, scheduler = deepspeed.initialize(
                model=raw_model,
                optimizer=None,
                model_parameters=model_parameters,
                lr_scheduler=sched_factory,
                config=ds_config,
                args={"local_rank": int(os.environ.get("LOCAL_RANK", "-1"))},
                dist_init_required=True,
            )
        finally:
            if _ds_set_flags_orig is not None:
                deepspeed.set_optimizer_flags = _ds_set_flags_orig

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

    def moving_average(self, model, model_ema, beta=0.992, device="cpu"):
        self.time_steps["ema"] += 1
        if self.time_steps["ema"] % self.accumulated_gradient == 0 or self.use_dynamic_batch:
            with torch.no_grad():
                for param, param_ema in zip(model.parameters(), model_ema.parameters()):
                    if param.requires_grad:
                        if self.stage != 3:
                            data = param.data.to(device)
                            param_ema.data.copy_((1 - beta) * data + beta * param_ema.data)
                        else:
                            # TODO: use prefiltering for efficiency
                            params_to_fetch = _z3_params_to_fetch([param, param_ema])
                            with deepspeed.zero.GatheredParameters(params_to_fetch, enabled=len(params_to_fetch) > 0):
                                data = param.data.to(device)
                                param_ema.data.copy_((1 - beta) * data + beta * param_ema.data)

    def load_model(
        self,
        model: nn.Module,
        path: str,
        map_location="cpu",
        strict: bool = False,
        key_replace_fn=None,
    ) -> None:
        unwrapped_model = self._unwrap_model(model)
        state_dict = torch.load(path, map_location=map_location)
        if key_replace_fn:
            state_dict = key_replace_fn(state_dict)
        unwrapped_model.load_state_dict(state_dict, strict=strict)

    def save_model(self, model: nn.Module, tokenizer, output_dir, **kwargs) -> None:
        if self.is_rank_0():
            os.makedirs(output_dir, exist_ok=True)

        # save model weights for ZeRO2/3
        model_to_save = self._unwrap_model(model)

        # gather parameters
        if self.args.ds.zero_stage > 2 or self.args.ds.tensor_parallel_size > 1:
            output_state_dict = (
                model.model._consolidated_16bit_state_dict()
                if isinstance(model, Actor)
                else model._consolidated_16bit_state_dict()
            )
        else:
            from deepspeed.checkpoint.utils import clone_tensors_for_torch_save

            output_state_dict = clone_tensors_for_torch_save(model_to_save.state_dict())

        if self.is_rank_0():
            state_dict_keys = set(model_to_save.state_dict().keys())
            output_state_dict_keys = set(output_state_dict.keys())

            # corner case for tie_word_embeddings, such as Qwen2-0.5B
            if getattr(model_to_save.config, "tie_word_embeddings", False) and "lm_head.weight" in state_dict_keys:
                state_dict_keys.remove("lm_head.weight")

            assert state_dict_keys.issubset(
                output_state_dict_keys
            ), f"mismatch keys {output_state_dict_keys.symmetric_difference(state_dict_keys)}"

            # only save peft weights https://github.com/microsoft/DeepSpeed/issues/4295
            if isinstance(model_to_save, PeftModel):
                model_to_save.save_pretrained(output_dir, **kwargs)
                if self.ds_tensor_parallel_size > 1 or self.stage == 3:
                    torch.save(
                        get_peft_model_state_dict(model_to_save, output_state_dict),
                        os.path.join(output_dir, "adapter_model.bin"),
                    )
                    filename = os.path.join(output_dir, "adapter_model.safetensors")
                    if os.path.exists(filename):
                        os.remove(filename)
            else:
                # save model
                model_to_save.save_pretrained(output_dir, state_dict=output_state_dict, **kwargs)

            # save config
            output_config_file = os.path.join(output_dir, "config.json")
            model_to_save.config.to_json_file(output_config_file)
            # save tokenizer
            tokenizer.save_pretrained(output_dir)

        del output_state_dict
        # Explicitly release memory
        import gc

        gc.collect()

        torch_dist_barrier_and_cuda_sync()

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

    def _get_ckpt_metric_path(self, ckpt_dir):
        return os.path.join(ckpt_dir, self.CKPT_METRIC_FILENAME)

    def _write_ckpt_metric(self, ckpt_dir, metric_value, metric_key=None):
        os.makedirs(ckpt_dir, exist_ok=True)
        metric_path = self._get_ckpt_metric_path(ckpt_dir)
        with open(metric_path, "w") as f:
            json.dump({"metric_key": metric_key, "metric_value": metric_value}, f, indent=2, sort_keys=True)

    def _read_ckpt_metric(self, ckpt_dir):
        metric_path = self._get_ckpt_metric_path(ckpt_dir)
        if not os.path.exists(metric_path):
            return None

        try:
            with open(metric_path, "r") as f:
                payload = json.load(f)
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
            self.print(f"Warning: failed to read checkpoint metric from {metric_path}: {exc}")
            return None

        if not isinstance(payload, dict):
            return None

        metric_value = payload.get("metric_value")
        if metric_value is None:
            return None

        try:
            return float(metric_value)
        except (TypeError, ValueError):
            self.print(f"Warning: invalid checkpoint metric in {metric_path}: {metric_value}")
            return None

    def save_ckpt(
        self,
        model,
        save_dir,
        tag=None,
        max_num=3,
        max_mem=1000,
        client_state=None,
        save_latest=True,
        metric_value=None,
        metric_key=None,
    ):
        assert isinstance(model, deepspeed.DeepSpeedEngine)
        client_state = client_state or {}
        is_best = tag is not None and tag.startswith("best")

        if self.is_rank_0():
            os.makedirs(save_dir, exist_ok=True)

            # Remove old best checkpoints when saving a new best.
            if is_best:
                for d in os.listdir(save_dir):
                    if d.startswith("best") and d != tag and os.path.isdir(os.path.join(save_dir, d)):
                        old_best = os.path.join(save_dir, d)
                        shutil.rmtree(old_best)
                        self.print(f"Removed old best checkpoint {old_best}")

            # Rotate old checkpoints to stay within max_num / max_mem limits.
            # Best checkpoints are protected from eviction and excluded from max_num counting.
            max_size_bytes = max_mem * 1024**3
            while True:
                all_subdirs = [
                    (os.path.join(save_dir, d), os.path.getmtime(os.path.join(save_dir, d)))
                    for d in os.listdir(save_dir)
                    if os.path.isdir(os.path.join(save_dir, d))
                ]
                regular_subdirs = [
                    (path, mtime) for path, mtime in all_subdirs if not os.path.basename(path).startswith("best")
                ]

                total_size = sum(
                    os.path.getsize(os.path.join(dirpath, f))
                    for subdir, _ in all_subdirs
                    for dirpath, _, filenames in os.walk(subdir)
                    for f in filenames
                )

                # +1 accounts for the checkpoint about to be saved; best ckpts don't count toward max_num
                overflow_num = max(0, len(regular_subdirs) - max_num + 1) if not is_best else 0
                overflow_mem = total_size > max_size_bytes
                if overflow_num == 0 and not overflow_mem:
                    break

                # Build eviction candidates from regular checkpoints only.
                # Sort: no-metric first (least informative), then worst metric, then oldest.
                candidates = sorted(
                    [(path, self._read_ckpt_metric(path), mtime) for path, mtime in regular_subdirs],
                    key=lambda item: (
                        item[1] is not None,  # None-metric first (delete first)
                        item[1] if item[1] is not None else float("-inf"),  # then worst metric
                        item[2],  # then oldest
                    ),
                )
                if not candidates:
                    break

                delete_dir, delete_metric, _ = candidates[0]
                reason = f"metric={delete_metric}" if delete_metric is not None else "no metric (oldest)"
                if os.path.exists(delete_dir):
                    shutil.rmtree(delete_dir)
                    self.print(f"Deleted checkpoint {delete_dir} ({reason})")

        torch_dist_barrier_and_cuda_sync()
        model.save_checkpoint(save_dir, tag=tag, client_state=client_state, save_latest=save_latest)

        # Write metric after successful save to avoid orphaned metric files on crash.
        if self.is_rank_0() and tag is not None:
            self._write_ckpt_metric(os.path.join(save_dir, tag), metric_value, metric_key=metric_key)

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
