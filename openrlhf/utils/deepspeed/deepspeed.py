import os
import shutil
from abc import ABC
from collections import defaultdict
from datetime import timedelta
from typing import List, Tuple, Union

import deepspeed
import torch
import torch.nn as nn
import torch.optim as optim
import transformers
import transformers.modeling_flash_attention_utils
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from peft import PeftModel, get_peft_model_state_dict
from torch import distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.optim import Optimizer
from torchdata.stateful_dataloader import StatefulDataLoader

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

ModelOptimPair = Tuple[nn.Module, Optimizer]
ModelOrModelOptimPair = Union[nn.Module, ModelOptimPair]


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
        bf16=True,
        args=None,
    ) -> None:
        super().__init__()

        self.args = args
        self.stage = zero_stage
        self.train_batch_size = train_batch_size
        self.micro_train_batch_size = micro_train_batch_size
        self.bf16 = bf16
        self.seed = seed
        self.full_determinism = full_determinism
        self.max_norm = max_norm

        self.adam_offload = getattr(args, "adam_offload", False)
        self.zpg = getattr(args, "zpg", 1)
        self.use_ds_universal_ckpt = getattr(args, "use_ds_universal_ckpt", False)
        self.grad_accum_dtype = getattr(args, "grad_accum_dtype", None)
        self.overlap_comm = getattr(args, "overlap_comm", False)
        self.deepcompile = getattr(args, "deepcompile", False)
        self.ds_tensor_parallel_size = getattr(args, "ds_tensor_parallel_size", 1)
        self.ring_attn_size = getattr(self.args, "ring_attn_size", 1)
        self.use_dynamic_batch = getattr(self.args, "use_dynamic_batch", False)

        if self.ds_tensor_parallel_size > 1:
            assert deepspeed.version >= "0.16.4", "DeepSpeed version must be >= 0.16.4 for tensor parallel training"
            assert bf16, "BF16 is required for tensor parallel training"

        self.is_rlhf = False
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

        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
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

        from ring_flash_attn import substitute_hf_flash_attn

        self.ring_head_stride = getattr(self.args, "ring_head_stride", 1)
        substitute_hf_flash_attn(self.ring_attn_group, self.ring_head_stride)

    @property
    def ring_attn_group(self):
        return get_ring_attn_group()

    def create_optimizer(self, model, **kwargs) -> Optimizer:
        if isinstance(model, Actor):
            model = model.model
        # Optimizer
        AdamOptimizer = DeepSpeedCPUAdam if self.adam_offload else FusedAdam
        optim_params = get_optimizer_grouped_parameters(model, kwargs["weight_decay"])
        optim = AdamOptimizer(optim_params, **kwargs)
        return optim

    def backward(self, loss: torch.Tensor, model: nn.Module, optimizer: optim.Optimizer, **kwargs) -> None:
        if isinstance(model, Actor):
            model = model.model
        model.backward(loss)

    def optimizer_step(
        self,
        optimizer: optim.Optimizer,
        model: nn.Module,
        scheduler,
        name="model",
        **kwargs,
    ) -> None:
        if isinstance(model, Actor):
            model = model.model
        model.step()

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
        )

    def _unwrap_model(self, model) -> nn.Module:
        if isinstance(model, Actor):
            return self._unwrap_model(model.model)
        elif hasattr(model, "module"):
            return model.module
        else:
            return model

    def prepare(
        self, *models_or_model_optim_pairs: ModelOrModelOptimPair, is_rlhf=False
    ) -> Union[List[ModelOrModelOptimPair], ModelOrModelOptimPair]:
        ret = []
        self.is_rlhf = is_rlhf
        for arg in models_or_model_optim_pairs:
            if isinstance(arg, tuple):
                assert len(arg) == 3, f'Expect (model, optimizer, scheduler) pair, got a tuple with size "{len(arg)}"'
                if arg[0] is not None:
                    ret.append(self._ds_init_train_model(*arg))
                else:
                    ret.append((None, None, None))
            else:
                ret.append(self._ds_init_eval_model(arg))

        return ret[0] if len(ret) == 1 else ret

    def _ds_init_train_model(self, model, optim, scheduler):
        is_actor = isinstance(model, Actor)
        ds_config = self.get_ds_train_config(is_actor)

        if self.ds_tensor_parallel_size > 1:
            tp_model = deepspeed.tp_model_init(
                model=model.model if is_actor else model, tp_size=self.ds_tensor_parallel_size, dtype=torch.bfloat16
            )
            if is_actor:
                model.model = tp_model
            else:
                model = tp_model

        engine, optim, _, scheduler = deepspeed.initialize(
            model=model.model if is_actor else model,
            optimizer=optim,
            lr_scheduler=scheduler,
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

    def get_ds_train_config(self, is_actor):
        # DS Config
        ds_config = get_train_ds_config(
            offload=False,
            adam_offload=self.adam_offload,
            stage=self.stage,
            bf16=self.bf16,
            max_norm=self.max_norm,
            zpg=self.zpg,
            grad_accum_dtype=self.grad_accum_dtype,
            overlap_comm=self.overlap_comm,
            use_ds_universal_ckpt=self.use_ds_universal_ckpt,
            deepcompile=self.deepcompile,
            tensor_parallel_size=self.ds_tensor_parallel_size,
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
            if is_actor:
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
            bf16=self.bf16,
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
        if self.args.zero_stage > 2 or self.args.ds_tensor_parallel_size > 1:
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

    def save_ckpt(self, model, save_dir, tag=None, max_num=3, max_mem=1000, client_state={}, save_latest=True):
        assert isinstance(model, deepspeed.DeepSpeedEngine)
        if self.is_rank_0():
            os.makedirs(save_dir, exist_ok=True)
            MAX_SIZE = max_mem * 1024**3  # Convert GB to bytes

            while True:
                subdirs = sorted(
                    [
                        (os.path.join(save_dir, d), os.path.getmtime(os.path.join(save_dir, d)))
                        for d in os.listdir(save_dir)
                        if os.path.isdir(os.path.join(save_dir, d))
                    ],
                    key=lambda x: x[1],
                )
                total_size = sum(
                    os.path.getsize(os.path.join(dirpath, f))
                    for subdir, _ in subdirs
                    for dirpath, _, filenames in os.walk(subdir)
                    for f in filenames
                )

                if len(subdirs) >= max_num or total_size > MAX_SIZE:
                    oldest_dir = subdirs[0][0]
                    if os.path.exists(oldest_dir):
                        shutil.rmtree(oldest_dir)
                        self.print(f"Deleted oldest ckpt {oldest_dir}")
                else:
                    break

        torch_dist_barrier_and_cuda_sync()
        model.save_checkpoint(save_dir, tag=tag, client_state=client_state, save_latest=save_latest)

        # Explicitly release memory
        import gc

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
            raise Exception(f"[deepspeed] failed to resume from checkpoint {load_dir}")
        return load_path, states
