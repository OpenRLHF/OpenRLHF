from abc import ABC, abstractmethod
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from typing import Any, List, Tuple, Union

ModelOptimPair = Tuple[nn.Module, Optimizer]
ModelOrModelOptimPair = Union[nn.Module, ModelOptimPair]


class Strategy(ABC):
    """
        Base class for training strategies.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def backward(self, loss: torch.Tensor, model: nn.Module, optimizer: Optimizer, **kwargs) -> None:
        pass

    @abstractmethod
    def optimizer_step(self, optimizer: Optimizer, model, scheduler, name='model', **kwargs) -> None:
        pass

    @abstractmethod
    def setup_distributed(self) -> None:
        pass

    @abstractmethod
    def setup_model(self, model: nn.Module) -> nn.Module:
        pass

    @abstractmethod
    def create_optimizer(self, model, **kwargs) -> Optimizer:
        pass

    @abstractmethod
    def setup_optimizer(self, optimizer: Optimizer, model: nn.Module) -> Optimizer:
        pass

    @abstractmethod
    def setup_scheduler(self, scheduler) -> Optimizer:
        pass

    @abstractmethod
    def setup_dataloader(self, replay_buffer, batch_size: int, pin_memory: bool = False, shuffle = True, collate_fn=None) -> DataLoader:
        pass

    def model_init_context(self):
        return nullcontext()

    @abstractmethod
    def _unwrap_model(self, model: nn.Module) -> nn.Module:
        pass
    
    @abstractmethod
    def prepare(
        self, *models_or_model_optim_pairs: ModelOrModelOptimPair, is_rlhf=False
    ) -> Union[List[ModelOrModelOptimPair], ModelOrModelOptimPair]:
        pass

    @abstractmethod
    def save_model(self, model: nn.Module, path: str, only_rank0: bool = False) -> None:
        pass

    @abstractmethod
    def load_model(self, model: nn.Module, path: str, map_location: Any = None, strict: bool = False, key_replace_fn=None) -> None:
        pass
    
    @abstractmethod
    def save_hf_format(self, model, tokenizer, output_dir):
        pass

    @abstractmethod
    def save_checkpoint(self, model, optimizer, scheduler, sampler, path: str, only_rank0: bool = False) -> None:
        pass

    @abstractmethod
    def load_checkpoint(self, model, optimizer, scheduler, sampler, path: str, map_location: Any = None) -> None:
        pass
    
    @abstractmethod
    def moving_average(self, model, model_ema, beta=0.992, device=None):
        pass
    
    @abstractmethod
    def all_reduce(self, data, op='mean'):
        pass

    @abstractmethod
    def all_gather(self, data):
        pass

    @abstractmethod
    def print(self, *msg):
        pass
    
    @abstractmethod
    def is_rank_0(self) -> bool:
        pass

