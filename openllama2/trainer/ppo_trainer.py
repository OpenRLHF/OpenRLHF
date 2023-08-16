import math
from abc import ABC
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from openllama2.models import Actor, Critic, GPTLMLoss, PolicyLoss, ValueLoss
from openllama2.models.utils import masked_mean

from .ppo_utils import (AdaptiveKLController, Experience, FixedKLController,
                        NaiveExperienceMaker, NaiveReplayBuffer)

SHOW_TIME_DETAILS = False

class PPOTrainer(ABC):
    """
        Trainer for PPO algorithm.

    Args:
        strategy (Strategy): the strategy to use for training
        actor (Actor): the actor model in ppo algorithm
        critic (Critic): the critic model in ppo algorithm
        reward_model (nn.Module): the reward model in rlhf algorithm to make reward of sentences
        initial_model (Actor): the initial model in rlhf algorithm to generate reference logits to limit the update of actor
        actor_optim (Optimizer): the optimizer to use for actor model
        critic_optim (Optimizer): the optimizer to use for critic model
        kl_coef (float, defaults to 0.1): the coefficient of kl divergence loss
        train_batch_size (int, defaults to 8): the batch size to use for training
        buffer_limit (int, defaults to 0): the max_size limitaiton of replay buffer
        buffer_cpu_offload (bool, defaults to True): whether to offload replay buffer to cpu
        eps_clip (float, defaults to 0.2): the clip coefficient of policy loss
        value_clip (float, defaults to 0.4): the clip coefficient of value loss
        experience_batch_size (int, defaults to 8): the batch size to use for experience generation
        max_epochs (int, defaults to 1): the number of epochs of training process
        tokenier (Callable, optional): the tokenizer to use for tokenizing the input
        sample_replay_buffer (bool, defaults to False): whether to sample from replay buffer
        dataloader_pin_memory (bool, defaults to True): whether to pin memory for data loader
        callbacks (List[Callback], defaults to []): the callbacks to call during training process
        generate_kwargs (dict, optional): the kwargs to use while model generating
    """

    def __init__(self,
                 strategy,
                 actor: Actor,
                 critic: Critic,
                 reward_model: nn.Module,
                 initial_model: Actor,
                 ema_model: Actor,
                 actor_optim: Optimizer,
                 critic_optim: Optimizer,
                 actor_scheduler,
                 critic_scheduler,
                 ema_beta: float = 0.992,
                 init_kl_coef: float = 0.001,
                 kl_target: float = None,
                 kl_horizon: int = 10000,
                 ptx_coef: float = 0,
                 micro_train_batch_size: int = 8,
                 buffer_limit: int = 0,
                 buffer_cpu_offload: bool = True,
                 eps_clip: float = 0.2,
                 value_clip: float = 0.2,
                 micro_rollout_batch_size: int = 8,
                 gradient_checkpointing: bool = False,
                 max_epochs: int = 1,
                 max_norm: float = 1.0,
                 tokenizer: Optional[Callable[[Any], dict]] = None,
                 prompt_max_len: int = 128, 
                 dataloader_pin_memory: bool = True,
                 **generate_kwargs) -> None:
        super().__init__()
        self.strategy = strategy
        self.micro_rollout_batch_size = micro_rollout_batch_size
        self.max_epochs = max_epochs
        self.tokenizer = tokenizer
        self.generate_kwargs = generate_kwargs
        self.dataloader_pin_memory = dataloader_pin_memory
        self.max_norm = max_norm
        self.ptx_coef = ptx_coef
        self.micro_train_batch_size = micro_train_batch_size
        self.kl_target = kl_target
        self.prompt_max_len = prompt_max_len
        self.ema_beta = ema_beta
        self.gradient_checkpointing = gradient_checkpointing
        
        self.actor = actor
        self.critic = critic
        self.ema_model = ema_model
        self.actor_optim = actor_optim
        self.critic_optim = critic_optim
        self.actor_scheduler = actor_scheduler
        self.critic_scheduler = critic_scheduler

        self.actor_loss_fn = PolicyLoss(eps_clip)
        self.critic_loss_fn = ValueLoss(value_clip)
        self.ptx_loss_fn = GPTLMLoss()

        if self.kl_target:
            self.kl_ctl = AdaptiveKLController(init_kl_coef, kl_target, kl_horizon)
        else:
            self.kl_ctl = FixedKLController(init_kl_coef)

        self.experience_maker = NaiveExperienceMaker(actor, critic, reward_model, initial_model, self.kl_ctl, strategy)
        self.replay_buffer = NaiveReplayBuffer(micro_train_batch_size, buffer_limit, buffer_cpu_offload)

        if self.gradient_checkpointing:
            self.actor.gradient_checkpointing_enable()
            self.critic.gradient_checkpointing_enable()

    def fit(self, prompts_dataloader, pretrain_dataloader, num_episodes: 1, rollout_batch_size: 1024) -> None:
        self.prompts_dataloader = prompts_dataloader
        self.pretrain_dataloader = pretrain_dataloader

        # tokenizer
        def tokenize_fn(texts):
            batch = self.tokenizer(texts, return_tensors='pt', max_length=self.prompt_max_len, 
                                   padding=True, truncation=True)
            return {k: v.to(torch.cuda.current_device()) for k, v in batch.items()}

        update_timesteps = rollout_batch_size // self.strategy.world_size * self.micro_rollout_batch_size
        time_step = 0

        for episode in range(num_episodes):
            if isinstance(self.prompts_dataloader.sampler, DistributedSampler):
                self.prompts_dataloader.sampler.set_epoch(episode)
            pbar = tqdm(self.prompts_dataloader,
                    desc=f'Episode [{episode+1}/{num_episodes}]',
                    disable=not self.strategy.is_rank_0())
            
            for rand_prompts in pbar:
                inputs = tokenize_fn(rand_prompts)
                experience = self.experience_maker.make_experience(**inputs, **self.generate_kwargs)
                self.replay_buffer.append(experience)

                time_step = (time_step + 1) % update_timesteps
                if time_step % update_timesteps == 0:
                    self.replay_buffer.normalize('advantages', self.strategy)
                    status = self.ppo_train()
                    self.replay_buffer.clear() 

                    self.kl_ctl.update(status['kl'], rollout_batch_size)
                    status['k_coef'] = self.kl_ctl.value
                    self.strategy.print(status)

    def ppo_train(self):
        # replay buffer may be empty at first, we should rebuild at each training
        dataloader = DataLoader(self.replay_buffer,
                          batch_size=self.replay_buffer.sample_batch_size,
                          shuffle=True,
                          drop_last=False,
                          pin_memory=self.dataloader_pin_memory,
                          collate_fn=self.replay_buffer.collate_fn)
        device = torch.cuda.current_device()

        status_list = []
        status_mean = {}
        for epoch in range(self.max_epochs):
            pbar = tqdm(dataloader, desc=f'Train epoch [{epoch+1}/{self.max_epochs}]', disable=not self.strategy.is_rank_0())
            for experience in pbar:
                experience.to_device(device)
                status = self.training_step(experience)

                # for DP
                # weighted mean for kl
                status['kl'] *= status['glen']
                status = self.strategy.all_reduce(status)
                status['kl'] /= status['glen']

                status_list.append(status)
                pbar.set_postfix(status)

        if status_list:
            status_mean = status_list[0]
            for m in status_list[1:]:
                for k, v in m.items():
                    status_mean[k] += v
            for k in status_mean.keys():
                status_mean[k] /= len(status_list)
        return status_mean

    def training_step(self, experience: Experience) -> Dict[str, float]:
        self.actor.train()
        self.critic.train()

        num_actions = experience.action_mask.size(1)
        for _ in tqdm(range(1), desc=f'Actor forward', disable=not SHOW_TIME_DETAILS or not self.strategy.is_rank_0()):
            action_log_probs = self.actor(experience.sequences, num_actions, attention_mask=experience.attention_mask)
            actor_loss = raw_actor_loss = self.actor_loss_fn(action_log_probs,
                                            experience.action_log_probs,
                                            experience.advantages,
                                            action_mask=experience.action_mask)
            # ptx loss
            if self.pretrain_dataloader is not None:
                data = next(self.pretrain_dataloader)
                inputs = data[1].squeeze(1).to(torch.cuda.current_device())
                attention_mask = data[2].squeeze(1).to(torch.cuda.current_device())
                label = torch.where(inputs.eq(self.tokenizer.pad_token_id), self.ptx_loss_fn.IGNORE_INDEX, inputs)

                ptx_log_probs = self.actor(inputs, attention_mask=attention_mask, return_output=True)['logits']
                ptx_loss = self.ptx_loss_fn(ptx_log_probs, label)
                actor_loss = ptx_loss * self.ptx_coef + raw_actor_loss
            
        for _ in tqdm(range(1), desc=f'Actor backward', disable=not SHOW_TIME_DETAILS or not self.strategy.is_rank_0()):
            self.strategy.backward(actor_loss, self.actor, self.actor_optim)
            self.strategy.optimizer_step(self.actor_optim, self.actor, self.actor_scheduler, name="actor")

            if self.ema_model:
                self.strategy.moving_average(self.actor, self.ema_model, self.ema_beta, 'cpu')

        for _ in tqdm(range(1), desc=f'Critic forward', disable=not SHOW_TIME_DETAILS or not self.strategy.is_rank_0()):
            values = self.critic(experience.sequences,
                                action_mask=experience.action_mask,
                                attention_mask=experience.attention_mask)
            critic_loss = self.critic_loss_fn(values,
                                            experience.values,
                                            experience.returns,
                                            action_mask=experience.action_mask)
            
        for _ in tqdm(range(1), desc=f'Critic backward', disable=not SHOW_TIME_DETAILS or not self.strategy.is_rank_0()):
            self.strategy.backward(critic_loss, self.critic, self.critic_optim)
            self.strategy.optimizer_step(self.critic_optim, self.critic, self.critic_scheduler, name="critic")

        # status
        status = {'pg': raw_actor_loss.item(), 
                'cri': critic_loss.item(),
                'vals':  masked_mean(values, experience.action_mask, dim=(0, 1)).item(),
                }
        if self.pretrain_dataloader is not None:
            status['ptx'] = ptx_loss.item()
        for k, v in experience.info.items():
            if k == 'kl':
                status[k] = ((v * experience.info['glen']).sum() / experience.info['glen'].sum()).item()
            else:
                status[k] = v.mean().item()
        return status

