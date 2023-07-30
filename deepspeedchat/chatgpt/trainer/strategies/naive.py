import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer
from chatgpt.models import Actor
from torch.utils.data import DataLoader
from torch.optim import AdamW
from peft import PeftModel

from .base import Strategy
import random
import numpy as np

import os
from typing import Any, List, Tuple, Union
from collections import defaultdict

ModelOptimPair = Tuple[nn.Module, Optimizer]
ModelOrModelOptimPair = Union[nn.Module, ModelOptimPair]

class NaiveStrategy(Strategy):
    """
        Strategy for single GPU. No parallelism is used.
    """

    def __init__(
        self,
        seed: int = 42,
        max_norm: float = 1.0,
        accumulated_gradient = 1,
        ) -> None:
        super().__init__()
        self.seed = seed
        self.max_norm = max_norm
        self.accumulated_gradient = accumulated_gradient
        self.time_steps = defaultdict(int)

        self.set_seed(seed)
        self.setup_distributed()

    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def backward(self, loss: torch.Tensor, model: nn.Module, optimizer: optim.Optimizer, **kwargs) -> None:
        (loss / self.accumulated_gradient).backward()

    def optimizer_step(self, optimizer: optim.Optimizer, model, scheduler, name='model', **kwargs) -> None:
        self.time_steps[name] += 1
        if self.time_steps[name] % self.accumulated_gradient == 0:
            nn.utils.clip_grad_norm_(model.parameters(), self.max_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

    def setup_distributed(self) -> None:
        self.world_size = 1

    def _unwrap_model(self, model: nn.Module) -> nn.Module:
        """Useful for saving state dict. As actor is wrapped by Actor class again in `prepare()`, we should unwrap it before saving.

        Args:
            model (nn.Module): an actor or a critic
        """
        if isinstance(model, Actor):
            return model.model
        return model

    def prepare(
        self, *models_or_model_optim_pairs: ModelOrModelOptimPair, is_rlhf=False
    ) -> Union[List[ModelOrModelOptimPair], ModelOrModelOptimPair]:
        rets = []
        for arg in models_or_model_optim_pairs:
            if isinstance(arg, tuple):
                assert len(arg) == 3, f'Expect (model, optimizer, scheduler) pair, got a tuple with size "{len(arg)}"'
                model, optimizer, scheduler = arg
                model = self.setup_model(model).to(torch.cuda.current_device())
                optimizer = self.setup_optimizer(optimizer, model)
                rets.append((model, optimizer, scheduler))
            elif isinstance(arg, nn.Module):
                model = self.setup_model(arg)
                if not getattr(model, 'is_ema', None):
                    model.to(torch.cuda.current_device())
                rets.append(model)
            else:
                raise RuntimeError(f'Expect model or (model, optimizer, scheduler) pair, got {type(arg)}')

        if len(rets) == 1:
            return rets[0]
        return rets

    def setup_model(self, model: nn.Module) -> nn.Module:
        return model

    def create_optimizer(self, model, **kwargs) -> Optimizer:
        return AdamW(model.parameters(), **kwargs)

    def setup_optimizer(self, optimizer: optim.Optimizer, model: nn.Module) -> optim.Optimizer:
        return optimizer

    def setup_scheduler(self, scheduler):
        return scheduler

    def setup_dataloader(self, replay_buffer, batch_size: int, pin_memory: bool = False, shuffle=True, collate_fn=None) -> DataLoader:
        return DataLoader(
            replay_buffer,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True,
            collate_fn=collate_fn,
            pin_memory=pin_memory)

    def save_model(self, model: nn.Module, path: str, only_rank0: bool = False) -> None:
        model_to_save = self._unwrap_model(model)
        if isinstance(model_to_save, PeftModel):
            model_to_save = model_to_save.merge_and_unload()
        save_dict = model_to_save.state_dict()
        torch.save(save_dict, path)

    def load_model(self, model: nn.Module, path: str, map_location: Any = 'cpu', strict: bool = False, key_replace_fn=None) -> None:
        unwrapped_model = self._unwrap_model(model)
        state_dict = torch.load(path, map_location=map_location)
        if key_replace_fn:
            state_dict=key_replace_fn(state_dict)
        unwrapped_model.load_state_dict(state_dict, strict=strict)

    def save_hf_format(self, model, tokenizer, output_dir):
        # used to save huggingface format, so we can use it for hf.from_pretrained
        model_to_save = self._unwrap_model(model)
        if isinstance(model_to_save, PeftModel):
            model_to_save = model_to_save.merge_and_unload()
        CONFIG_NAME = "config.json"
        WEIGHTS_NAME = "pytorch_model.bin"
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        save_dict = model_to_save.state_dict()
        torch.save(save_dict, output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(output_dir)

    def save_checkpoint(self, model, optimizer, scheduler, sampler, path: str, only_rank0: bool = False) -> None:
        pass

    def load_checkpoint(self, model, optimizer, scheduler, sampler, path: str, map_location: Any = None) -> None:
        pass

    def moving_average(self, model, model_ema, beta=0.992, device='cpu'):
        self.time_steps['ema'] += 1
        if self.time_steps['ema'] % self.accumulated_gradient == 0:
            with torch.no_grad():
                for param, param_ema in zip(model.parameters(),
                                            model_ema.parameters()):
                    data = param.data.to(device)
                    param_ema.data.copy_((1 - beta) * data + beta * param_ema.data)

    def all_reduce(self, data, op='mean'):
        return data

    def all_gather(self, data):
        return data

    def print(self, *msg):
        print(*msg)

    def is_rank_0(self) -> bool:
        return True