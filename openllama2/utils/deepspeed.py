import os
import random
from abc import ABC
from typing import List, Tuple, Union
from collections import defaultdict

import deepspeed
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torch import distributed as dist
from torch.optim import Optimizer

from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from peft import PeftModel

from openllama2.models import Actor

from .deepspeed_utils import (_z3_params_to_fetch, get_eval_ds_config,
                              get_optimizer_grouped_parameters,
                              get_train_ds_config)

ModelOptimPair = Tuple[nn.Module, Optimizer]
ModelOrModelOptimPair = Union[nn.Module, ModelOptimPair]

class DeepspeedStrategy(ABC):
    """
        The strategy for training with Accelerator.
    """

    def __init__(
            self,
            seed: int = 42,
            max_norm: float = 0.0,
            micro_train_batch_size = 1,
            train_batch_size = 1,
            zero_stage = 2,
            max_out_tokens=512,
            inference_tp_size=1,
            bf16=True,
            args = None,
            ) -> None:
        super().__init__()

        self.args = args
        self.stage = zero_stage
        self.train_batch_size = train_batch_size
        self.max_out_tokens = max_out_tokens
        self.micro_train_batch_size = micro_train_batch_size
        self.inference_tp_size = inference_tp_size
        self.bf16 = bf16
        self.adam_offload = args.adam_offload
        self.is_rlhf = False
        self.zpg = args.zpg
        self.seed = seed
        self.max_norm = max_norm
        self.time_steps = defaultdict(int)

        self.set_seed(seed)
        self.setup_distributed()

    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def setup_distributed(self) -> None:
        if "SLURM_PROCID" in os.environ: # for slurm
            self.args.local_rank = int(os.environ['LOCAL_RANK'])

        if self.args.local_rank != -1:
            torch.cuda.set_device(self.args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        deepspeed.init_distributed()
        self.world_size = dist.get_world_size()
        self.accumulated_gradient = self.train_batch_size // self.micro_train_batch_size // self.world_size

    def create_optimizer(self, model, **kwargs) -> Optimizer:
        if isinstance(model, Actor):
            model = model.model
        # Optimizer
        AdamOptimizer = DeepSpeedCPUAdam if self.adam_offload else FusedAdam
        optim_params = get_optimizer_grouped_parameters(model, kwargs['weight_decay'])
        optim = AdamOptimizer(optim_params, **kwargs)
        return optim

    def backward(self, loss: torch.Tensor, model: nn.Module, optimizer: optim.Optimizer, **kwargs) -> None:
        if isinstance(model, Actor):
            model = model.model
        model.backward(loss)

    def optimizer_step(self, optimizer: optim.Optimizer, model: nn.Module, scheduler, name='model', **kwargs) -> None:
        if isinstance(model, Actor):
            model = model.model
        model.step()

    def setup_dataloader(self, replay_buffer, batch_size: int, pin_memory: bool = False, shuffle=True, collate_fn=None):
        # DDP only mode, replay buffers on each rank are different.
        sampler = DistributedSampler(replay_buffer,
                                        num_replicas=dist.get_world_size(),
                                        rank=dist.get_rank(),
                                        shuffle=shuffle,
                                        seed=self.seed,
                                        drop_last=True)
        return DataLoader(
            replay_buffer,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=True,
            collate_fn=collate_fn,
            pin_memory=pin_memory)

    def _unwrap_model(self, model) -> nn.Module:
        if isinstance(model, Actor):
            return self._unwrap_model(model.model)
        elif hasattr(model, 'module'):
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
                ret.append(self.ds_init_train_model(*arg))
            else:
                ret.append(self.ds_init_eval_model(arg))

        return ret[0] if len(ret) == 1 else ret

    def ds_init_train_model(self, model, optim, scheduler):
        is_actor = isinstance(model, Actor)
        stage = self.stage
        # ZeRO3-based GPT generation is very slow in RLHF
        if self.is_rlhf and is_actor and stage == 3 and self.inference_tp_size <= 1:
            stage = 2

        # DS Config
        ds_config = get_train_ds_config(
            offload=False,
            adam_offload=self.adam_offload,
            stage=stage,
            bf16=self.bf16,
            max_norm=self.max_norm, 
            # hybrid_engine does not support a lot of models
            enable_hybrid_engine=is_actor and self.inference_tp_size > 1 and stage == 3, 
            pin_parameters=True,
            inference_tp_size=self.inference_tp_size,
            tp_gather_partition_size=4,
            max_out_tokens=self.max_out_tokens,
            zpg=self.zpg)
        # dummy batch size
        ds_config['train_micro_batch_size_per_gpu'] =  self.micro_train_batch_size 
        ds_config['train_batch_size'] =  self.train_batch_size

        engine, optim, _, scheduler = deepspeed.initialize(model=model.model if is_actor else model,
                                                optimizer=optim,
                                                lr_scheduler=scheduler,
                                                config=ds_config,
                                                args={"local_rank": self.args.local_rank},
                                                dist_init_required=True)
        if is_actor:
            model.model = engine
        else:
            model = engine
        return model, optim, scheduler

    def ds_init_eval_model(self, model):
        is_actor = isinstance(model, Actor)
        stage = self.stage
        offload = False
        # No gradients
        if stage != 3:
            stage = 0
        # Offload ema model
        if getattr(model, 'is_ema', None):
            offload = True
            stage = 0

        # DS Config
        ds_config = get_eval_ds_config(offload=offload,
                                       stage=stage,
                                       bf16=self.bf16,
                                       enable_hybrid_engine=is_actor and self.inference_tp_size > 1 and stage == 3,
                                       inference_tp_size=self.inference_tp_size,
                                       tp_gather_partition_size=self.inference_tp_size,
                                       max_out_tokens=self.max_out_tokens)
        ds_config['train_micro_batch_size_per_gpu'] =  self.micro_train_batch_size 
        ds_config['train_batch_size'] =  self.train_batch_size
                            
        engine, *_ = deepspeed.initialize(model=model.model if is_actor else model, 
                                              args={"local_rank": self.args.local_rank},
                                              config=ds_config, 
                                              dist_init_required=True)
        if is_actor:
            model.model = engine
        else:
            model = engine
        return model

    def moving_average(self, model, model_ema, beta=0.992, device='cpu'):
        self.time_steps['ema'] += 1
        if self.time_steps['ema'] % self.accumulated_gradient == 0:
            with torch.no_grad():
                for param, param_ema in zip(model.parameters(), model_ema.parameters()):
                    if param.requires_grad:
                        if self.stage != 3:
                            data = param.data.to(device)
                            param_ema.data.copy_((1 - beta) * data + beta * param_ema.data)
                        else:
                            # TODO: use prefiltering for efficiency
                            params_to_fetch = _z3_params_to_fetch([param, param_ema])
                            with deepspeed.zero.GatheredParameters(params_to_fetch, 
                                                                    enabled=len(params_to_fetch) > 0):
                                data = param.data.to(device)
                                param_ema.data.copy_((1 - beta) * data + beta * param_ema.data)

    def load_model(self, model: nn.Module, path: str, map_location = 'cpu', strict: bool = False, key_replace_fn=None) -> None:
        unwrapped_model = self._unwrap_model(model)
        state_dict = torch.load(path, map_location=map_location)
        if key_replace_fn:
            state_dict=key_replace_fn(state_dict)
        unwrapped_model.load_state_dict(state_dict, strict=strict)

    def save_model(self, model: nn.Module, path: str, only_rank0: bool = True) -> None:
        model_to_save = self._unwrap_model(model)
        if isinstance(model_to_save, PeftModel):
            model_to_save = model_to_save.merge_and_unload()

        if self.stage != 3 or self.is_rlhf:
            if self.is_rank_0():
                save_dict = model_to_save.state_dict()
                torch.save(save_dict, path)
        else:
            output_state_dict = {}
            # gather parameters
            for k, v in model_to_save.named_parameters():
                params_to_fetch = _z3_params_to_fetch([v])
                with deepspeed.zero.GatheredParameters(params_to_fetch, 
                                                        enabled=len(params_to_fetch) > 0):
                    vv = v.data.cpu()
                    if self.is_rank_0():
                        output_state_dict[k] = vv
            if self.is_rank_0():
                for k, v in model_to_save.named_buffers():
                    vv = v.data.cpu()
                    output_state_dict[k] = vv
                torch.save(output_state_dict, path)

    def save_hf_format(self, model, tokenizer, output_dir):
        # used to save huggingface format, so we can use it for hf.from_pretrained
        CONFIG_NAME = "config.json"
        WEIGHTS_NAME = "pytorch_model.bin"
        # save model weights for ZeRO2/3
        self.save_model(model,os.path.join(output_dir, WEIGHTS_NAME))
        # save config
        model_to_save = self._unwrap_model(model)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        model_to_save.config.to_json_file(output_config_file)
        # save tokenizer
        tokenizer.save_vocabulary(output_dir)

    def all_reduce(self, data, op='mean'):
        assert op in ('mean', 'max', 'sum')
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
            if op == 'mean':
                data /= self.world_size
            dist.all_reduce(data, op=dist.ReduceOp.MAX if op == 'max' else dist.ReduceOp.SUM)
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
        return dist.get_rank() == 0
