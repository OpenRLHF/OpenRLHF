from typing import Union, Tuple, List

import os
import torch
import torch.nn as nn
import torch.optim as optim
from chatgpt.models import Actor
from torch.optim import Optimizer
from chatgpt.trainer.strategies import DDPStrategy
from torch import distributed as dist

import deepspeed
from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam
from .utils import get_optimizer_grouped_parameters, get_eval_ds_config, get_train_ds_config, _z3_params_to_fetch
from peft import PeftModel


ModelOptimPair = Tuple[nn.Module, Optimizer]
ModelOrModelOptimPair = Union[nn.Module, ModelOptimPair]

class DeepspeedStrategy(DDPStrategy):
    """
        The strategy for training with Accelerator.
    """

    def __init__(
            self,
            seed: int = 42,
            max_norm: float = 0.0,
            accumulated_gradient = 1,
            train_batch_size = 1,
            zero_stage = 2,
            max_out_tokens=512,
            inference_tp_size=1,
            bf16=True,
            args = None,
            ) -> None:
        self.args = args
        super().__init__(seed, max_norm, accumulated_gradient)
        
        self.stage = zero_stage
        self.train_batch_size=train_batch_size
        self.max_out_tokens = max_out_tokens
        self.inference_tp_size = inference_tp_size
        self.bf16 = bf16
        self.adam_offload = False
        self.is_rlhf = False

    def model_init_context(self):
        return super().model_init_context()

    def setup_distributed(self) -> None:
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        deepspeed.init_distributed()
        if self.args.local_rank != -1:
            torch.cuda.set_device(self.args.local_rank)
        self.world_size = dist.get_world_size()

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

    def _unwrap_model(self, model) -> nn.Module:
        if isinstance(model, Actor):
            return self._unwrap_model(model.model)
        elif hasattr(model, 'module'):
            return model.module
        else:
            return model

    def _init_trainable_model(self, model, optim, scheduler):
        is_actor = isinstance(model, Actor)
        stage = self.stage
        # ZeRO3-based GPT generation is very slow in RLHF
        if self.is_rlhf and is_actor and stage == 3 and self.inference_tp_size <= 1:
            stage = 2

        # DS Config
        ds_config = get_train_ds_config(
            offload=False,
            stage=stage,
            bf16=self.bf16,
            max_norm=self.max_norm, 
            # hybrid_engine does not support a lot of models
            enable_hybrid_engine=is_actor and self.inference_tp_size > 1 and stage == 3, 
            pin_parameters=True,
            inference_tp_size=self.inference_tp_size,
            tp_gather_partition_size=self.inference_tp_size,
            max_out_tokens=self.max_out_tokens,
            zpg=8)
        # dummy batch size
        ds_config['train_micro_batch_size_per_gpu'] =  self.train_batch_size
        ds_config['gradient_accumulation_steps'] =  self.accumulated_gradient

        engine, optim, _, scheduler = deepspeed.initialize(model=model.model if is_actor else model,
                                                optimizer=optim,
                                                lr_scheduler=scheduler,
                                                config=ds_config,
                                                args=self.args,
                                                dist_init_required=True)
        if is_actor:
            model.model = engine
        else:
            model = engine
        return model, optim, scheduler

    def _init_freeze_model(self, model):
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
        ds_config['train_micro_batch_size_per_gpu'] =  self.train_batch_size
        ds_config['gradient_accumulation_steps'] =  self.accumulated_gradient
                            
        engine, *_ = deepspeed.initialize(model=model.model if is_actor else model, 
                                              args=self.args,
                                              config=ds_config, 
                                              dist_init_required=True)
        if is_actor:
            model.model = engine
        else:
            model = engine
        return model

    def prepare(
        self, *models_or_model_optim_pairs: ModelOrModelOptimPair, is_rlhf=False
    ) -> Union[List[ModelOrModelOptimPair], ModelOrModelOptimPair]:
        ret = []
        self.is_rlhf = is_rlhf
        for arg in models_or_model_optim_pairs:
            if isinstance(arg, tuple):
                assert len(arg) == 3, f'Expect (model, optimizer, scheduler) pair, got a tuple with size "{len(arg)}"'
                ret.append(self._init_trainable_model(*arg))
            else:
                ret.append(self._init_freeze_model(arg))

        return ret[0] if len(ret) == 1 else ret

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
