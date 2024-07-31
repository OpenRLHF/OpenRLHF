import math
import os
from typing import Dict, Optional

import ray
import torch
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers.trainer import get_scheduler

from openrlhf.models import Actor, get_llm_for_sequence_regression
from openrlhf.trainer import PPOTrainer
from openrlhf.trainer.ppo_utils import Experience
from openrlhf.utils import DeepspeedStrategy, blending_datasets, get_tokenizer

from .launcher import BasePPORole


class CriticPPOTrainer(PPOTrainer):
    def ppo_train(self):
        # replay buffer may be empty at first, we should rebuild at each training
        dataloader = DataLoader(
            self.replay_buffer,
            batch_size=self.replay_buffer.sample_batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=self.dataloader_pin_memory,
            collate_fn=self.replay_buffer.collate_fn,
        )
        device = torch.cuda.current_device()

        status_list = []
        status_mean = {}
        for epoch in range(self.max_epochs):
            pbar = tqdm(
                dataloader,
                desc=f"Train epoch [{epoch + 1}/{self.max_epochs}]",
                disable=not self.strategy.is_rank_0(),
            )
            for experience in pbar:
                experience.to_device(device)
                status = self.training_step(experience)

                # for DP
                status = self.strategy.all_reduce(status)

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
        return self.training_step_critic(experience)


@ray.remote(num_gpus=1)
class CriticModelRayActor(BasePPORole):
    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain, max_steps):
        args = strategy.args

        self._setup_distributed(strategy)
        critic = get_llm_for_sequence_regression(
            pretrain,
            "critic",
            normalize_reward=strategy.args.normalize_reward,
            use_flash_attention_2=strategy.args.flash_attn,
            bf16=strategy.args.bf16,
            load_in_4bit=strategy.args.load_in_4bit,
            lora_rank=strategy.args.lora_rank,
            lora_alpha=strategy.args.lora_alpha,
            target_modules=strategy.args.target_modules,
            lora_dropout=strategy.args.lora_dropout,
            ds_config=strategy.get_ds_train_config(is_actor=False),
            value_head_prefix=strategy.args.value_head_prefix,
            init_value_head=strategy.args.pretrain == strategy.args.critic_pretrain,
        )
        strategy.print(critic)
        strategy.print("reward normalization status: {}".format(strategy.args.normalize_reward))
        strategy.print("mean: {}, std {}".format(critic.mean, critic.std))

        # configure tokenizer
        if strategy.args.save_value_network:
            self.tokenizer = get_tokenizer(
                pretrain, critic, "left", strategy, use_fast=not strategy.args.disable_fast_tokenizer
            )

        # configure optimizer
        critic_optim = strategy.create_optimizer(
            critic, lr=args.critic_learning_rate, betas=args.adam_betas, weight_decay=args.l2
        )

        # configure scheduler
        critic_scheduler = get_scheduler(
            "cosine_with_min_lr",
            critic_optim,
            num_warmup_steps=math.ceil(max_steps * 0.03),
            num_training_steps=max_steps,
            scheduler_specific_kwargs={"min_lr": args.critic_learning_rate * 0.1},
        )

        if args.gradient_checkpointing:
            critic.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
            )

        # prepare models/optimizers...
        self.critic, self.critic_optim, self.critic_scheduler = strategy.prepare(
            (critic, critic_optim, critic_scheduler),
            is_rlhf=True,
        )

        # load checkpoint
        if args.load_checkpoint and os.path.exists(os.path.join(args.ckpt_path, "_actor")):
            ckpt_path = os.path.join(args.ckpt_path, "_critic")
            strategy.load_ckpt(self.critic, ckpt_path)
            strategy.print(f"Loaded the checkpoint: {ckpt_path}")

        # configure Trainer
        # only use wandb at actor model
        strategy.args.use_wandb = False
        self.trainer = CriticPPOTrainer(
            strategy,
            actor=None,
            critic=self.critic,
            reward_model=None,
            initial_model=None,
            ema_model=None,
            actor_optim=None,
            critic_optim=self.critic_optim,
            actor_scheduler=None,
            critic_scheduler=self.critic_scheduler,
            max_epochs=args.max_epochs,
            micro_train_batch_size=args.micro_train_batch_size,
            micro_rollout_batch_size=args.micro_rollout_batch_size,
            gradient_checkpointing=args.gradient_checkpointing,
            prompt_max_len=args.prompt_max_len,
            value_clip=args.value_clip,
            eps_clip=args.eps_clip,
        )

    def forward(
        self,
        sequences: torch.LongTensor,
        action_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generates critic values."""
        device = torch.cuda.current_device()
        self.critic.eval()
        with torch.no_grad():
            value = self.critic(sequences.to(device), action_mask.to(device), attention_mask.to(device))
        self.critic.train()  # reset model state
        return value.to("cpu")

    def append(self, experience):
        """Append experience to replay buffer."""
        self.trainer.replay_buffer.append(experience)

    def fit(self):
        """Train critic model with the replay buffer."""
        torch.cuda.empty_cache()
        self.critic.train()
        status = self.trainer.ppo_train()
        self.trainer.replay_buffer.clear()
        torch.cuda.empty_cache()
        return status

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()

    def save_model(self):
        args = self.strategy.args

        # save model checkpoint after fitting on only rank0
        self.strategy.save_model(
            self.critic,
            self.tokenizer,
            args.save_path + "_critic",
        )

    def save_checkpoint(self, tag):
        args = self.strategy.args
        self.strategy.save_ckpt(
            self.critic, os.path.join(args.ckpt_path, "_critic"), tag, args.max_ckpt_num, args.max_ckpt_mem
        )
