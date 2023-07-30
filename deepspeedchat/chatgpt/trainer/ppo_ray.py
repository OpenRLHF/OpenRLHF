from typing import Any, Callable, Dict, List, Optional, Union
import math

import torch
from torch import Tensor
import torch.nn as nn
from chatgpt.experience_maker import Experience, NaiveExperienceMaker
from chatgpt.models import Actor, Critic, PolicyLoss, ValueLoss, GPTLMLoss
from chatgpt.replay_buffer import NaiveReplayBuffer
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler

from .ppo import PPOTrainer
from .strategies import Strategy, DDPStrategy
from .utils import AdaptiveKLController, FixedKLController

from chatgpt.models.utils import masked_mean
from tqdm import tqdm


class RayPPOTrainer(PPOTrainer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def fit(self, prompts_dataloader, pretrain_dataloader, num_episodes: int = 1, update_timesteps: int = 32) -> None:
        pass

    def _ppo_train(self):
        pass

    def _training_step(self, experience: Experience) -> Dict[str, float]:
        pass