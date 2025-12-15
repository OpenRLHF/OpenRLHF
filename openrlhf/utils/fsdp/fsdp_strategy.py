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

from torch.distributed.fsdp import CPUOffloadPolicy, FSDPModule, MixedPrecisionPolicy, fully_shard
from torch.optim import Optimizer
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import enable_full_determinism, set_seed

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

        self.fsdp2_offload = getattr(args, "fsdp2_offload", "none")
        self.fsdp2_cpu_offload_pin_memory = self._coerce_bool(
            getattr(args, "fsdp2_cpu_offload_pin_memory", True),
            default=True,
        )
        self.fsdp2_reshard_after_forward = self._coerce_bool(
            getattr(args, "fsdp2_reshard_after_forward", True),
            default=True,
        )

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

        self.accumulated_gradient = max(
            1, self.train_batch_size // self.micro_train_batch_size // max(1, self.world_size)
        )
        # Dynamic batch mode manages stepping explicitly (see PPO trainers); disable fixed accumulation.
        if getattr(self.args, "use_dynamic_batch", False):
            self.accumulated_gradient = 1

    @property
    def ring_attn_group(self):
        return get_ring_attn_group()

    def create_optimizer(self, model, **kwargs) -> Optimizer:
        if isinstance(model, Actor):
            model = model.model
        return optim.AdamW(model.parameters(), **kwargs)

    def backward(self, loss: torch.Tensor, model: nn.Module, optimizer: optim.Optimizer, **kwargs) -> None:
        # Match DeepSpeed semantics: average gradients over accumulation steps.
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
        micro_step_key = f"optim_micro_step_{name}"
        self.time_steps[micro_step_key] += 1
        accumulation_steps = max(1, int(getattr(self, "accumulated_gradient", 1)))
        if self.time_steps[micro_step_key] % accumulation_steps != 0:
            return

        # Unwrap Actor to get the underlying FSDPModule
        unwrapped = self._unwrap_model(model)
        if self.max_norm and self.max_norm > 0:
            devices = {p.device.type for p in unwrapped.parameters() if p.grad is not None}
            if "cpu" in devices:
                # Avoid DTensor all_reduce on CPU backends; skip clipping when offloaded.
                self.print("Warning: Gradient clipping is skipped when using FSDP2 CPU offload.")
            elif hasattr(unwrapped, "clip_grad_norm_"):
                unwrapped.clip_grad_norm_(self.max_norm)
            else:
                torch.nn.utils.clip_grad_norm_(unwrapped.parameters(), self.max_norm)
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
        """Unwrap Actor wrapper, return the inner model."""
        if isinstance(model, Actor):
            return self._unwrap_model(model.model)
        return model

    def _wrap_train_model(self, model: nn.Module) -> nn.Module:
        """Wrap model with FSDP2 fully_shard."""
        try:
            self._maybe_fully_shard_children(model)
            if not isinstance(model, FSDPModule):
                model = fully_shard(
                    model,
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
        offload_mode = (self.fsdp2_offload or "none").lower()
        if offload_mode == "none":
            return None
        if offload_mode == "cpu":
            return CPUOffloadPolicy(pin_memory=bool(self.fsdp2_cpu_offload_pin_memory))
        raise ValueError(f"Unknown fsdp2_offload mode: {self.fsdp2_offload}")

    def _build_mixed_precision_policy(self) -> Optional["MixedPrecisionPolicy"]:
        if not self.bf16:
            return None
        # Use float32 for reduce_dtype to maintain gradient precision during all-reduce
        return MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=None)

    @staticmethod
    def _move_optimizer_state(optimizer: optim.Optimizer, device: torch.device) -> None:
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

    def _maybe_fully_shard_children(self, module: nn.Module) -> None:
        layer_cls_to_wrap = getattr(module, "_no_split_modules", None)

        if layer_cls_to_wrap:
            # Policy-based wrapping for HF models
            for name, child in module.named_modules():
                if child.__class__.__name__ in layer_cls_to_wrap:
                    fully_shard(
                        child,
                        reshard_after_forward=self.fsdp2_reshard_after_forward,
                        offload_policy=self._offload_policy,
                        mp_policy=self._mp_policy,
                    )
                # Also wrap Embeddings if not tied (optional, but good practice)
                elif (
                    isinstance(child, nn.Embedding)
                    and hasattr(module, "config")
                    and not module.config.tie_word_embeddings
                ):
                    fully_shard(
                        child,
                        reshard_after_forward=self.fsdp2_reshard_after_forward,
                        offload_policy=self._offload_policy,
                        mp_policy=self._mp_policy,
                    )
        else:
            # Fallback: shallow wrapping for non-HF models
            for name, child in module.named_children():
                if isinstance(child, FSDPModule):
                    continue
                if isinstance(child, nn.ModuleList):
                    new_list = []
                    changed = False
                    for sub in child:
                        if isinstance(sub, nn.Module) and not isinstance(sub, FSDPModule):
                            fully_shard(
                                sub,
                                reshard_after_forward=self.fsdp2_reshard_after_forward,
                                offload_policy=self._offload_policy,
                                mp_policy=self._mp_policy,
                            )
                            new_list.append(sub)
                            changed = True
                        else:
                            new_list.append(sub)
                    if changed:
                        setattr(module, name, nn.ModuleList(new_list))
                    continue
                if not any(p.requires_grad for p in child.parameters(recurse=True)):
                    continue
                fully_shard(
                    child,
                    reshard_after_forward=self.fsdp2_reshard_after_forward,
                    offload_policy=self._offload_policy,
                    mp_policy=self._mp_policy,
                )

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
                module = self._wrap_train_model(module)
                module = module.to(torch.cuda.current_device())
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
                module = module.to(torch.cuda.current_device())
                if is_actor:
                    model.model = module
                    ret.append(model)
                else:
                    ret.append(module)
        return ret[0] if len(ret) == 1 else ret

    def offload_states(self, model, optimizer=None):
        """Offload model and optimizer states to CPU."""
        self._unwrap_model(model).cpu()
        if optimizer is not None:
            self._move_optimizer_state(optimizer, torch.device("cpu"))
        torch.cuda.empty_cache()
        if dist.is_initialized():
            dist.barrier()

    def reload_states(self, model, optimizer=None):
        """Reload model and optimizer states to GPU."""
        device = torch.device("cuda", torch.cuda.current_device())
        self._unwrap_model(model).to(device)
        if optimizer is not None:
            self._move_optimizer_state(optimizer, device)
        torch.cuda.synchronize()  # Ensure async operations complete
        if dist.is_initialized():
            dist.barrier()

    def moving_average(self, model, model_ema, beta=0.992, device="cpu"):
        self.time_steps["ema"] += 1
        if self.time_steps["ema"] % max(1, self.accumulated_gradient) == 0:
            # Use _unwrap_model to get the underlying model
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
        """Update EMA model parameters using full state dict."""
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
        state_dict = torch.load(path, map_location=map_location)
        if key_replace_fn:
            state_dict = key_replace_fn(state_dict)

        # Use _unwrap_model to get the underlying model
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
        if self.is_rank_0():
            os.makedirs(output_dir, exist_ok=True)

        fsdp_model = self._unwrap_model(model)

        model_state = get_model_state_dict(
            model=fsdp_model,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
        if self.is_rank_0():
            fsdp_model.save_pretrained(output_dir, state_dict=model_state, **kwargs)
            fsdp_model.config.to_json_file(os.path.join(output_dir, "config.json"))
            tokenizer.save_pretrained(output_dir)
        if dist.is_initialized():
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
                data = data / self.world_size
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
            is_tensor = True
            if not isinstance(data, torch.Tensor):
                data = torch.tensor([data])
                is_tensor = False
            is_cpu_tensor = data.device.type == "cpu"

            ret = [torch.zeros_like(data).to(torch.cuda.current_device()) for _ in range(self.world_size)]
            dist.all_gather(ret, data.to(torch.cuda.current_device()))
            result = torch.cat(ret).cpu() if is_cpu_tensor else torch.cat(ret)
            # If original input is not a tensor, return list to maintain type consistency
            if not is_tensor:
                return result.tolist()
            return result

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
