import os
from abc import ABC
from collections import defaultdict
from datetime import timedelta
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import enable_full_determinism, set_seed

try:
    from torch.distributed.checkpoint.state_dict import (  # type: ignore[attr-defined]
        StateDictOptions,
        get_model_state_dict,
        set_model_state_dict,
    )
except ImportError:  # pragma: no cover - older torch versions
    StateDictOptions = None  # type: ignore
    get_model_state_dict = None  # type: ignore
    set_model_state_dict = None  # type: ignore

from openrlhf.models import Actor
from openrlhf.models.ring_attn_utils import get_ring_attn_group, set_ring_attn_group
from openrlhf.utils.distributed_sampler import DistributedSampler

ModelOptimPair = Tuple[nn.Module, Optimizer]
ModelOrModelOptimPair = Union[nn.Module, ModelOptimPair]


class FSDPStrategy(ABC):
    def __init__(
        self,
        seed: int = 42,
        full_determinism: bool = False,
        max_norm: float = 0.0,
        micro_train_batch_size=1,
        train_batch_size=1,
        bf16=True,
        args=None,
    ) -> None:
        super().__init__()

        self.args = args
        self.train_batch_size = train_batch_size
        self.micro_train_batch_size = micro_train_batch_size
        self.bf16 = bf16
        self.seed = seed
        self.full_determinism = full_determinism
        self.max_norm = max_norm

        self.fsdp_offload = getattr(args, "fsdp_offload", "none")
        self.fsdp_cpu_offload_pin_memory = self._coerce_bool(
            getattr(args, "fsdp_cpu_offload_pin_memory", True),
            default=True,
        )
        self.fsdp_reshard_after_forward = self._coerce_bool(
            getattr(args, "fsdp_reshard_after_forward", True),
            default=True,
        )

        self.is_rlhf = False
        self.time_steps = defaultdict(int)
        torch_version = version.parse(torch.__version__.split("+")[0])
        required_version = version.parse("2.4.0")
        if fully_shard is None or FSDPModule is None or torch_version < required_version:
            raise RuntimeError(
                "FSDP2 backend requires PyTorch >= 2.4 with the fully_shard API. "
                f"Detected torch=={torch.__version__}. Please upgrade PyTorch or use --dist_backend deepspeed."
            )
        if StateDictOptions is None or get_model_state_dict is None or set_model_state_dict is None:
            raise RuntimeError(
                "torch.distributed.checkpoint.state_dict APIs are required for FSDP2 checkpointing. "
                "Please install a recent PyTorch build with distributed.checkpoint enabled."
            )

        self._offload_policy: Optional["CPUOffloadPolicy"] = self._build_offload_policy()
        self._mp_policy: Optional["MixedPrecisionPolicy"] = self._build_mixed_precision_policy()

    def setup_distributed(self, timeout=timedelta(minutes=60)) -> None:
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

        self.accumulated_gradient = self.train_batch_size // self.micro_train_batch_size // max(1, self.world_size)

    @property
    def ring_attn_group(self):
        return get_ring_attn_group()

    def create_optimizer(self, model, **kwargs) -> Optimizer:
        if isinstance(model, Actor):
            model = model.model
        return optim.AdamW(model.parameters(), **kwargs)

    def backward(self, loss: torch.Tensor, model: nn.Module, optimizer: optim.Optimizer, **kwargs) -> None:
        loss.backward()

    def optimizer_step(
        self,
        optimizer: optim.Optimizer,
        model: nn.Module,
        scheduler,
        name="model",
        **kwargs,
    ) -> None:
        if self.max_norm and self.max_norm > 0:
            if hasattr(model, "clip_grad_norm_"):
                model.clip_grad_norm_(self.max_norm)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
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
        if sampler is None and dist.is_initialized():
            num_replicas = dist.get_world_size()
            rank = dist.get_rank()

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
        if isinstance(model, Actor):
            return self._unwrap_model(model.model)
        if FSDPModule is not None and isinstance(model, FSDPModule):
            return model.module
        return model

    def _wrap_train_model(self, model: nn.Module) -> nn.Module:
        return self._wrap_train_model_fsdp2(model)

    @staticmethod
    def _coerce_bool(value, default=False):
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "y", "t"}:
                return True
            if lowered in {"false", "0", "no", "n", "f"}:
                return False
        return bool(value)

    def _build_offload_policy(self) -> Optional["CPUOffloadPolicy"]:
        offload_mode = (self.fsdp_offload or "none").lower()
        if offload_mode == "none":
            return None
        if offload_mode == "cpu":
            if CPUOffloadPolicy is None:
                raise RuntimeError(
                    "CPUOffloadPolicy is unavailable in this torch build. Please upgrade PyTorch to use fsdp_offload=cpu."
                )
            return CPUOffloadPolicy(pin_memory=bool(self.fsdp_cpu_offload_pin_memory))
        raise ValueError(f"Unknown fsdp_offload mode: {self.fsdp_offload}")

    def _build_mixed_precision_policy(self) -> Optional["MixedPrecisionPolicy"]:
        if not self.bf16:
            return None
        if MixedPrecisionPolicy is None:
            raise RuntimeError(
                "MixedPrecisionPolicy is unavailable in this torch build. Please upgrade PyTorch to use bf16 with FSDP."
            )
        return MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16
        )

    def _maybe_fully_shard_children(self, module: nn.Module) -> None:
        assert fully_shard is not None and FSDPModule is not None
        for name, child in module.named_children():
            if isinstance(child, FSDPModule):
                continue
            if isinstance(child, nn.ModuleList):
                new_list = []
                changed = False
                for sub in child:
                    if isinstance(sub, nn.Module) and not isinstance(sub, FSDPModule):
                        new_sub = fully_shard(
                            sub,
                            reshard_after_forward=self.fsdp_reshard_after_forward,
                            offload_policy=self._offload_policy,
                            mixed_precision=self._mp_policy,
                        )
                        new_list.append(new_sub)
                        changed = True
                    else:
                        new_list.append(sub)
                if changed:
                    setattr(module, name, nn.ModuleList(new_list))
                continue
            if not any(p.requires_grad for p in child.parameters(recurse=False)):
                continue
            new_child = fully_shard(
                child,
                reshard_after_forward=self.fsdp_reshard_after_forward,
                offload_policy=self._offload_policy,
                mixed_precision=self._mp_policy,
            )
            if new_child is not child:
                setattr(module, name, new_child)

    def _wrap_train_model_fsdp2(self, model: nn.Module) -> nn.Module:
        assert fully_shard is not None and FSDPModule is not None
        try:
            # shard large submodules first (e.g. transformer blocks) if possible
            self._maybe_fully_shard_children(model)
            if not isinstance(model, FSDPModule):
                model = fully_shard(
                    model,
                    reshard_after_forward=self.fsdp_reshard_after_forward,
                    offload_policy=self._offload_policy,
                    mixed_precision=self._mp_policy,
                )
            return model
        except Exception as exc:  # pragma: no cover - defensive fallback
            raise RuntimeError(
                "fully_shard() failed while constructing the FSDP2 model. "
                "Please double-check that the model supports composable FSDP."
            ) from exc

    def prepare(self, *models_or_model_optim_pairs: ModelOrModelOptimPair, is_rlhf=False):
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
                module = module.to(torch.cuda.current_device())
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
                module = module.to(torch.cuda.current_device())
                module = self._wrap_train_model(module)
                if is_actor:
                    model.model = module
                    ret.append(model)
                else:
                    ret.append(module)
        return ret[0] if len(ret) == 1 else ret

    def moving_average(self, model, model_ema, beta=0.992, device="cpu"):
        self.time_steps["ema"] += 1
        if self.time_steps["ema"] % max(1, self.accumulated_gradient) == 0:
            with torch.no_grad():
                for param, param_ema in zip(model.parameters(), model_ema.parameters()):
                    if not param.requires_grad:
                        continue
                    param_ema.data.mul_(beta)
                    param_ema.data.add_(param, alpha=1 - beta)

    def load_model(
        self,
        model: nn.Module,
        path: str,
        map_location="cpu",
        strict: bool = False,
        key_replace_fn=None,
    ) -> None:
        state_dict = torch.load(path, map_location=map_location)
        if key_replace_fn:
            state_dict = key_replace_fn(state_dict)

    def save_model(self, model: nn.Module, tokenizer, output_dir, **kwargs) -> None:
        if self.is_rank_0():
            os.makedirs(output_dir, exist_ok=True)

        wrapped_model = model.model if isinstance(model, Actor) else model

        if FSDPModule is not None and isinstance(wrapped_model, FSDPModule):
            model_state = get_model_state_dict(
                model=wrapped_model,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            )
            if self.is_rank_0():
                unwrapped = self._unwrap_model(model)
                unwrapped.save_pretrained(output_dir, state_dict=model_state, **kwargs)
                output_config_file = os.path.join(output_dir, "config.json")
                unwrapped.config.to_json_file(output_config_file)
                tokenizer.save_pretrained(output_dir)
            dist.barrier()
            return

        if self.is_rank_0():
            unwrapped = self._unwrap_model(model)
            cpu_state_dict = unwrapped.state_dict()
            unwrapped.save_pretrained(output_dir, state_dict=cpu_state_dict, **kwargs)
            output_config_file = os.path.join(output_dir, "config.json")
            unwrapped.config.to_json_file(output_config_file)
            tokenizer.save_pretrained(output_dir)

        dist.barrier()

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
                data = torch.tensor([data])
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
                data = torch.tensor([data])
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

    def save_ckpt(self, *args, **kwargs):
        return None

    def load_ckpt(self, *args, **kwargs):
        return None, {"consumed_samples": 0}

    def get_ds_train_config(self, is_actor):
        return None

    def get_ds_eval_config(self, offload=False):
        return None
