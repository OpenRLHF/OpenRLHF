import math
import os
from abc import ABC
from typing import Dict, Optional, Union

import ray
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.trainer import get_scheduler

from openrlhf.models import ValueLoss, get_llm_for_sequence_regression
from openrlhf.models.utils import masked_mean
from openrlhf.trainer.ppo_utils.experience_maker import Experience
from openrlhf.utils.fsdp2.strategy import FSDP2Strategy

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
        self.max_epochs = self.args.max_epochs

        self.replay_buffer = NaiveReplayBuffer(
            micro_train_batch_size,
            buffer_limit,
            buffer_cpu_offload,
            getattr(self.args, "packing_samples", False),
            self.args.use_dynamic_batch,
        )

        self.critic_loss_fn = ValueLoss(value_clip)

        # Mixtral 8x7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

    def ppo_train(self):
        # replay buffer may be empty at first, we should rebuild at each training
        if self.args.use_dynamic_batch:
            self.replay_buffer.setup_dynamic_batch(self.strategy)

        not_shuffle = (
            self.strategy.ring_attn_group is not None or self.args.fsdp2_tp_size > 1 or self.args.use_dynamic_batch
        )
        dataloader = DataLoader(
            self.replay_buffer,
            batch_size=self.replay_buffer.sample_batch_size,
            shuffle=not not_shuffle,
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
        loss = critic_loss + aux_loss * self.args.aux_loss_coef
        if self.args.use_dynamic_batch:
            loss = loss * self.replay_buffer.dynamic_loss_scale[step]

        self.strategy.backward(loss, self.critic, self.critic_optim, name="critic")
        if self.args.use_dynamic_batch:
            if self.replay_buffer.dynamic_optimizer_step[step]:
                self.strategy.optimizer_step(self.critic_optim, self.critic, self.critic_scheduler, name="critic")
        else:
            self.strategy.optimizer_step(self.critic_optim, self.critic, self.critic_scheduler, name="critic")

        # status
        status = {
            "critic_loss": critic_loss.detach().item(),
            "values": masked_mean(values, experience.action_mask).detach().item(),
            "critic_lr": self.critic_scheduler.get_last_lr()[0],
        }
        return status


@ray.remote(num_gpus=1)
class CriticModelActor(BaseModelActor):
    def init_model_from_pretrained(self, strategy: FSDP2Strategy, pretrain, max_steps):
        args = strategy.args
        self.disable_fsdp2_ckpt = args.disable_fsdp2_ckpt

        self._setup_distributed(strategy)
        init_value_head = strategy.args.model_name_or_path == strategy.args.critic_model_name_or_path
        critic = get_llm_for_sequence_regression(
            pretrain,
            "critic",
            normalize_reward=strategy.args.normalize_reward,
            attn_implementation=strategy.args.attn_implementation,
            torch_dtype=torch.float32,
            value_head_prefix=strategy.args.value_head_prefix,
            packing_samples=strategy.args.packing_samples,
        )
        strategy.print(critic)
        strategy.print("reward normalization status: {}".format(strategy.args.normalize_reward))
        strategy.print("mean: {}, std {}".format(critic.mean, critic.std))

        if args.gradient_checkpointing:
            critic.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
            )

        # Wrap/shard model(s) before building optimizer/scheduler (params become DTensor/sharded).
        critic = strategy.apply_parallelism(critic)
        strategy.load_hf_checkpoint(
            critic,
            pretrain,
            init_value_head=init_value_head,
            value_head_prefix=strategy.args.value_head_prefix,
        )

        # configure optimizer (after model wrapping for FSDP2)
        critic_optim = strategy.create_optimizer(
            critic, lr=args.critic_learning_rate, betas=args.adam_betas, weight_decay=args.l2
        )

        # configure scheduler
        critic_scheduler = get_scheduler(
            args.lr_scheduler,
            critic_optim,
            num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
            num_training_steps=max_steps,
            scheduler_specific_kwargs={"min_lr": args.critic_learning_rate * 0.1},
        )

        self.critic = critic
        self.critic_optim = critic_optim
        self.critic_scheduler = critic_scheduler

        # load checkpoint
        resume_from_path = getattr(args, "resume_from_path", None)
        if resume_from_path:
            actor_dir = os.path.join(resume_from_path, "dcp_checkpoint", "_actor")
            if not os.path.isdir(actor_dir):
                raise FileNotFoundError(
                    f"Invalid resume_from_path: expected actor checkpoint directory at {actor_dir}"
                )
            critic_dir = os.path.join(resume_from_path, "dcp_checkpoint", "_critic")
            if not os.path.isdir(critic_dir):
                raise FileNotFoundError(
                    f"Invalid resume_from_path: expected critic checkpoint directory at {critic_dir}"
                )
            strategy.print(f"Loading critic checkpoint: {critic_dir}")
            strategy.load_dcp_checkpoint(self.critic, critic_dir, optimizer=self.critic_optim, scheduler=self.critic_scheduler)

        # initial offload
        if strategy.args.fsdp2_enable_sleep:
            self.offload_states()

        # configure Trainer
        self.trainer = CriticPPOTrainer(
            strategy,
            critic=self.critic,
            critic_optim=self.critic_optim,
            critic_scheduler=self.critic_scheduler,
            micro_train_batch_size=args.micro_train_batch_size,
            value_clip=args.value_clip,
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

    def save_checkpoint(self, tag):
        if self.disable_fsdp2_ckpt:
            return False

        step_dir = os.path.join(self.strategy.dcp_ckpt_path, tag)
        self.strategy.save_dcp_checkpoint(
            self.critic,
            os.path.join(step_dir, "dcp_checkpoint", "_critic"),
            optimizer=self.critic_optim,
            scheduler=self.critic_scheduler,
        )
        return True

    def reload_states(self):
        self.strategy.reload_optimizer_states(self.critic_optim)

    def offload_states(self):
        self.strategy.offload_optimizer_states(self.critic_optim)

    def offload_model(self):
        """Offload model to CPU for rollout phase (hybrid engine mode)."""
        if isinstance(self.strategy, FSDP2Strategy):
            self.strategy.offload_model(self.critic)

    def reload_model(self):
        """Reload model to GPU for forward pass (hybrid engine mode)."""
        if isinstance(self.strategy, FSDP2Strategy):
            self.strategy.reload_model(self.critic)
