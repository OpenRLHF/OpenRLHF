import os
from abc import ABC
from collections import defaultdict
from datetime import timedelta
from typing import List, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP
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

        self.is_rlhf = False
        self.time_steps = defaultdict(int)

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
        # ring attention group is optional and matches existing behavior
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
        if isinstance(model, (FSDP, DDP)):
            return model.module
        return model

    def _wrap_train_model(self, model: nn.Module) -> nn.Module:
        # Auto wrap transformer blocks if possible; otherwise fall back to DDP
        try:
            auto_wrap = transformer_auto_wrap_policy
            model = FSDP(
                model,
                auto_wrap_policy=auto_wrap,
                device_id=torch.cuda.current_device(),
            )
            return model
        except Exception as e:
            self.print(f"FSDP auto-wrap failed, falling back to DDP: {e}")
            return DDP(model, device_ids=[torch.cuda.current_device()])

    def prepare(
        self, *models_or_model_optim_pairs: ModelOrModelOptimPair, is_rlhf=False
    ):
        ret = []
        self.is_rlhf = is_rlhf
        for arg in models_or_model_optim_pairs:
            if isinstance(arg, tuple):
                assert len(arg) == 3, f'Expect (model, optimizer, scheduler) pair, got a tuple with size "{len(arg)}"'
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
                module = DDP(module, device_ids=[torch.cuda.current_device()])
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
                    if param.requires_grad:
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
        state_dict = torch.load(path, map_location=map_location)
        if key_replace_fn:
            state_dict = key_replace_fn(state_dict)

    def save_model(self, model: nn.Module, tokenizer, output_dir, **kwargs) -> None:
        if self.is_rank_0():
            os.makedirs(output_dir, exist_ok=True)

        # Determine wrapped module
        wrapped_model = model.model if isinstance(model, Actor) else model

        # Gather a full state dict when using FSDP; fallback for DDP/nn.Module
        cpu_state_dict = None
        if isinstance(wrapped_model, FSDP):
            save_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(wrapped_model, StateDictType.FULL_STATE_DICT, save_cfg):
                cpu_state_dict = wrapped_model.state_dict()

        # Rank0 writes the HF checkpoint
        if self.is_rank_0():
            if cpu_state_dict is None:
                # DDP / non-sharded case
                cpu_state_dict = self._unwrap_model(model).state_dict()

            unwrapped = self._unwrap_model(model)
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

    # FSDP uses standard torch.save in trainers; DeepSpeed-like ckpt helpers are not provided here.
    def save_ckpt(self, *args, **kwargs):
        # Not supported: trainers should rely on save_model for HF checkpoints.
        return None

    def load_ckpt(self, *args, **kwargs):
        # Return a dummy state for compatibility with trainers expecting consumed_samples
        return None, {"consumed_samples": 0}

    # Compatibility shims for actor construction paths expecting DeepSpeed configs
    def get_ds_train_config(self, is_actor):
        return None

    def get_ds_eval_config(self, offload=False):
        return None
