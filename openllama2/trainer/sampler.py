import math
from abc import ABC
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from openllama2.models import Actor


class Sampler(ABC):
    def __init__(
        self,
        strategy,
        actor: Actor,
        reward_model: nn.Module,
        prompt_max_len: int = 128,
        dataloader_pin_memory: bool = True,
        **generate_kwargs,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.generate_kwargs = generate_kwargs
        self.dataloader_pin_memory = dataloader_pin_memory
        self.prompt_max_len = prompt_max_len

        self._wandb = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/epoch")
            wandb.define_metric("eval/*", step_metric="eval/epoch", step_sync=True)

    def fit(
        self,
        prompts_dataloader,
        pretrain_dataloader,
        num_episodes: 1,
        rollout_batch_size: 1024,
    ) -> None:
        self.prompts_dataloader = prompts_dataloader
        self.pretrain_dataloader = pretrain_dataloader

        # tokenizer
        def tokenize_fn(texts):
            batch = self.tokenizer(
                texts,
                return_tensors="pt",
                max_length=self.prompt_max_len,
                padding=True,
                truncation=True,
            )
            return {k: v.to(torch.cuda.current_device()) for k, v in batch.items()}

        update_timesteps = rollout_batch_size // (self.strategy.world_size * self.micro_rollout_batch_size)
        global_step = 0

        for episode in range(num_episodes):
            if isinstance(self.prompts_dataloader.sampler, DistributedSampler):
                self.prompts_dataloader.sampler.set_epoch(episode)
            pbar = tqdm(
                self.prompts_dataloader,
                desc=f"Episode [{episode+1}/{num_episodes}]",
                disable=not self.strategy.is_rank_0(),
            )

            for rand_prompts in pbar:
                inputs = tokenize_fn(rand_prompts)
                self.strategy.print(status)
                if self._wandb is not None and self.strategy.is_rank_0():
                    logs = {
                        "train/%s" % k: v
                        for k, v in {
                            **status,
                            "global_step": global_step // update_timesteps,
                        }.items()
                    }
                    self._wandb.log(logs)
