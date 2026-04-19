import os
from abc import ABC
from typing import Dict, Optional, Union

import ray
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from openrlhf.models import ValueLoss, get_llm_for_sequence_regression
from openrlhf.models.utils import masked_mean
from openrlhf.trainer.ppo_utils.experience import Experience
from openrlhf.utils import get_tokenizer
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.utils.deepspeed.deepspeed_utils import (
    offload_deepspeed_states,
    reload_deepspeed_states,
)

from ..ppo_utils import NaiveReplayBuffer
from .launcher import BaseModelActor


class CriticPPOTrainer(ABC):
    def __init__(
        self,
        strategy,
        critic: torch.nn.Module,
        critic_optim: Optimizer,
        critic_scheduler,
        micro_train_batch_size: int = 8,
        buffer_limit: int = 0,
        buffer_cpu_offload: bool = True,
        value_clip: float = 0.2,
        dataloader_pin_memory: bool = True,
        **kwargs,
    ):
        self.strategy = strategy
        self.args = strategy.args
        self.critic = critic
        self.critic_optim = critic_optim
        self.critic_scheduler = critic_scheduler
        self.micro_train_batch_size = micro_train_batch_size
        self.buffer_limit = buffer_limit
        self.buffer_cpu_offload = buffer_cpu_offload
        self.value_clip = value_clip
        self.dataloader_pin_memory = dataloader_pin_memory
        self.max_epochs = self.args.train.max_epochs

        self.replay_buffer = NaiveReplayBuffer(
            micro_train_batch_size,
            buffer_limit,
            buffer_cpu_offload,
            getattr(self.args.data, "packing_samples", False),
            self.args.train.dynamic_batch_enable,
        )

        self.critic_loss_fn = ValueLoss(value_clip)

        # Mixtral 8x7b
        self.aux_loss = self.args.actor.aux_loss_coef > 1e-8

    def ppo_train(self):
        # replay buffer may be empty at first, we should rebuild at each training
        if self.args.train.dynamic_batch_enable:
            self.replay_buffer.setup_dynamic_batch(self.strategy)

        should_shuffle = (
            self.strategy.ring_attn_group is None
            and self.args.ds.tensor_parallel_size <= 1
            and not self.args.train.dynamic_batch_enable
        )
        dataloader = DataLoader(
            self.replay_buffer,
            batch_size=self.replay_buffer.sample_batch_size,
            shuffle=should_shuffle,
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
            for step, experience in enumerate(pbar):
                experience.to_device(device)
                status = self.training_step(experience, step)

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

    def training_step(self, experience: Experience, step: int) -> Dict[str, float]:
        self.critic.train()

        sequences = experience.sequences
        old_values = experience.values
        returns = experience.returns
        action_mask = experience.action_mask
        packed_seq_lens = None
        attention_mask = experience.attention_mask

        # critic loss
        values, output = self.critic(
            sequences,
            action_mask=action_mask,
            attention_mask=attention_mask,
            return_output=True,
            ring_attn_group=self.strategy.ring_attn_group,
            values_allgather=True,
            packed_seq_lens=packed_seq_lens,
        )

        # loss function
        critic_loss = self.critic_loss_fn(
            values,
            old_values,
            returns,
            action_mask=experience.action_mask,
        )
        # mixtral
        if self.aux_loss:
            aux_loss = output.aux_loss
        else:
            aux_loss = 0
        loss = critic_loss + aux_loss * self.args.actor.aux_loss_coef
        if self.args.train.dynamic_batch_enable:
            loss = loss * self.replay_buffer.dynamic_loss_scale[step]

        self.strategy.backward(loss, self.critic, self.critic_optim)
        if self.args.train.dynamic_batch_enable:
            if self.replay_buffer.dynamic_optimizer_step[step]:
                self.strategy.optimizer_step(self.critic_optim, self.critic, self.critic_scheduler, name="critic")
        else:
            self.strategy.optimizer_step(self.critic_optim, self.critic, self.critic_scheduler, name="critic")

        # status
        status = {
            "critic_loss": critic_loss.detach().item(),
            "values": masked_mean(values, experience.action_mask).detach().item(),
            "critic_lr": self.critic_scheduler.get_last_lr()[0],
            "critic_grad_norm": self.strategy.get_grad_norm(self.critic),
        }
        return status


@ray.remote(num_gpus=1)
class CriticModelActor(BaseModelActor):
    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain, max_steps):
        args = strategy.args
        self.disable_ds_ckpt = args.ckpt.disable_ds

        self._setup_distributed(strategy)
        critic = get_llm_for_sequence_regression(
            pretrain,
            "critic",
            normalize_reward=strategy.args.reward.normalize_enable,
            attn_implementation=strategy.args.ds.attn_implementation,
            experts_implementation=strategy.args.ds.experts_implementation,
            param_dtype=strategy.args.ds.param_dtype,  # default: bf16
            load_in_4bit=strategy.args.ds.load_in_4bit,
            lora_rank=strategy.args.ds.lora.rank,
            lora_alpha=strategy.args.ds.lora.alpha,
            target_modules=strategy.args.ds.lora.target_modules,
            lora_dropout=strategy.args.ds.lora.dropout,
            ds_config=strategy.get_ds_train_config(is_actor=False),
            value_head_prefix=strategy.args.ds.value_head_prefix,
            init_value_head=strategy.args.actor.model_name_or_path == strategy.args.critic.model_name_or_path,
            packing_samples=strategy.args.ds.packing_samples,
        )
        strategy.print(critic)
        strategy.print("reward normalization status: {}".format(strategy.args.reward.normalize_enable))
        strategy.print("mean: {}, std {}".format(critic.mean, critic.std))

        # configure tokenizer (only when we plan to save the critic weights as HF)
        self.tokenizer = None
        if strategy.args.critic.save_value_network:
            self.tokenizer = get_tokenizer(
                pretrain, critic, "left", strategy, use_fast=not strategy.args.data.disable_fast_tokenizer
            )

        if args.actor.gradient_checkpointing_enable:
            critic.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": args.actor.gradient_checkpointing_reentrant}
            )

        # Critic reads its own args.critic.* sub-namespace.  Typical setup: actor may
        # use Muon but --critic.optim stays adam because value heads are essentially 1D.
        critic_cfg = dict(
            optim=args.critic.optim,
            muon=vars(args.critic.muon),
            adam=vars(args.critic.adam),
            lr_scheduler=args.critic.lr_scheduler,
            lr_warmup_ratio=args.critic.lr_warmup_ratio,
            min_lr_ratio=args.critic.min_lr_ratio,
            max_norm=args.critic.max_norm,
            scheduler_steps=max_steps,
        )
        self.critic, self.critic_optim, self.critic_scheduler = strategy.prepare((critic, critic_cfg))

        # load checkpoint
        ckpt_path = os.path.join(args.ckpt.path, "_critic")
        if args.ckpt.load_enable and os.path.exists(ckpt_path):
            strategy.print(f"Loading the checkpoint: {ckpt_path}")
            strategy.load_ckpt(self.critic, ckpt_path)

        # initial offload
        if strategy.args.ds.enable_sleep:
            self.offload_states()

        # configure Trainer
        self.trainer = CriticPPOTrainer(
            strategy,
            critic=self.critic,
            critic_optim=self.critic_optim,
            critic_scheduler=self.critic_scheduler,
            micro_train_batch_size=args.train.micro_batch_size,
            value_clip=args.critic.value_clip,
        )

    def forward(
        self,
        sequences: torch.LongTensor,
        action_mask: Optional[Union[int, list[int]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        packed_seq_lens=None,
    ) -> torch.Tensor:
        """Generates critic values."""
        device = torch.cuda.current_device()
        self.critic.eval()
        with torch.no_grad():
            value = self.critic(
                sequences.to(device),
                action_mask.to(device),
                attention_mask.to(device),
                ring_attn_group=self.strategy.ring_attn_group,
                values_allgather=True,
            )
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
        torch.cuda.synchronize()
        return status

    def save_model(self):
        args = self.strategy.args
        if self.tokenizer is None:
            # critic built without --critic.save_value_network; nothing to persist as HF weights
            return

        # save model checkpoint after fitting on only rank0
        self.strategy.save_model(
            self.critic,
            self.tokenizer,
            args.ckpt.output_dir + "_critic",
        )

    def save_checkpoint(self, tag, metric_value=None, metric_key=None):
        args = self.strategy.args
        if not self.disable_ds_ckpt:
            self.strategy.save_ckpt(
                self.critic,
                os.path.join(args.ckpt.path, "_critic"),
                tag,
                args.ckpt.max_num,
                args.ckpt.max_mem,
                metric_value=metric_value,
                metric_key=metric_key,
            )

    def reload_states(self):
        reload_deepspeed_states(self.critic)

    def offload_states(self):
        offload_deepspeed_states(self.critic)
