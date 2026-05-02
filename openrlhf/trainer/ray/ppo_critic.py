import gc
import os
from abc import ABC
from contextlib import ExitStack
from typing import Dict, Optional, Union

import ray
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from openrlhf.models import ValueLoss, get_llm_for_sequence_regression
from openrlhf.models.utils import masked_mean, split_moe_aux_loss
from openrlhf.trainer.ppo_utils.experience import Experience, get_model_parallel_size
from openrlhf.utils import get_tokenizer
from openrlhf.utils.fsdp import FsdpStrategy
from openrlhf.utils.fsdp.packing import cp_shard_sequence, pad_to_cp_multiple

from ..ppo_utils import NaiveReplayBuffer
from .launcher import BaseModelActor


def _cp_local_step_tensor(
    tensor: Optional[torch.Tensor],
    *,
    cp_group,
    cp_size: int,
    sequence_length: int,
    value: int | float | bool = 0,
) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    if cp_size <= 1:
        return tensor
    if tensor.shape[1] == sequence_length - 1:
        pad_shape = list(tensor.shape)
        pad_shape[1] = 1
        tensor = torch.cat([tensor, tensor.new_full(pad_shape, value)], dim=1)
    elif tensor.shape[1] != sequence_length:
        raise ValueError(
            f"CP local step tensor has length {tensor.shape[1]}, expected {sequence_length - 1} or {sequence_length}."
        )
    tensor = pad_to_cp_multiple(tensor, cp_size, seq_dim=1, value=value)
    return cp_shard_sequence(tensor, cp_group, seq_dim=1)


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
            dynamic_batch=self.args.train.dynamic_batch_enable,
        )

        self.critic_loss_fn = ValueLoss(value_clip)

        # MoE balancing loss.
        self.aux_loss = self.args.actor.aux_loss_coef > 1e-8

    def ppo_train(self):
        # replay buffer may be empty at first, we should rebuild at each training
        if self.args.train.dynamic_batch_enable:
            self.replay_buffer.setup_dynamic_batch(self.strategy)

        should_shuffle = get_model_parallel_size(self.args) <= 1 and not self.args.train.dynamic_batch_enable
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
            max_steps = len(dataloader)
            if not self.args.train.dynamic_batch_enable:
                # Only run complete accumulation windows; partial windows leave
                # gradients live because optimizer_step() has not stepped yet.
                accum_steps = self.strategy.accumulated_gradient
                remainder = max_steps % accum_steps
                if remainder:
                    max_steps -= remainder
                    self.strategy.print(
                        f"[Critic] dropping {remainder} trailing critic microbatches "
                        f"(< grad_accum={accum_steps}) to avoid partial gradients."
                    )
            for step, experience in enumerate(pbar):
                if step >= max_steps:
                    break
                experience.to_device(device)
                status = self.training_step(experience, step)
                status = self.strategy.all_reduce(status)
                status_list.append(status)
                pbar.set_postfix(status)

        if status_list:
            status_mean = {}
            for k in set().union(*(m.keys() for m in status_list)):
                vals = [m[k] for m in status_list if k in m]
                if k == "critic_lr":
                    status_mean[k] = vals[-1]
                else:
                    status_mean[k] = sum(vals) / len(vals)
        return status_mean

    def training_step(self, experience: Experience, step: int) -> Dict[str, float]:
        self.critic.train()

        sequences = experience.sequences
        old_values = experience.values
        returns = experience.returns
        action_mask = experience.action_mask
        packed_seq_lens = None
        attention_mask = experience.attention_mask
        loss_action_mask = action_mask

        # AutoModel CP train context must cover both forward and backward.
        cp_context_stack = ExitStack()

        # critic loss
        cp_local_loss = (
            getattr(self.critic, "cp_size", 1) > 1
            and not self.critic.packing_samples
            and os.environ.get("OPENRLHF_CP_LOCAL_VALUE_LOSS", "0") == "1"
        )
        batch_num_tokens = None
        loss_dp_size = 1
        if cp_local_loss:
            batch_num_tokens = self.strategy.global_token_count(action_mask)
            loss_dp_size = getattr(self.strategy, "dp_cp_size", self.strategy.dp_size)
            cp_group = self.critic.cp_mesh.get_group()
            cp_size = self.critic.cp_size
            sequence_length = sequences.shape[1]
            loss_action_mask = _cp_local_step_tensor(
                action_mask,
                cp_group=cp_group,
                cp_size=cp_size,
                sequence_length=sequence_length,
                value=False,
            )
            old_values = _cp_local_step_tensor(
                old_values,
                cp_group=cp_group,
                cp_size=cp_size,
                sequence_length=sequence_length,
                value=0.0,
            )
            returns = _cp_local_step_tensor(
                returns,
                cp_group=cp_group,
                cp_size=cp_size,
                sequence_length=sequence_length,
                value=0.0,
            )
        values, output = self.critic(
            sequences,
            action_mask=loss_action_mask,
            attention_mask=attention_mask,
            return_output=True,
            values_allgather=not cp_local_loss,
            cp_local_values=cp_local_loss,
            cp_context_stack=cp_context_stack,
            packed_seq_lens=packed_seq_lens,
        )

        # loss function
        critic_loss = self.critic_loss_fn(
            values,
            old_values,
            returns,
            action_mask=loss_action_mask,
            dp_size=loss_dp_size,
            batch_num_tokens=batch_num_tokens,
        )
        # mixtral
        aux_loss, _ = split_moe_aux_loss(output, self.aux_loss)
        loss = critic_loss + aux_loss * self.args.actor.aux_loss_coef
        if self.args.train.dynamic_batch_enable:
            loss = loss * self.replay_buffer.dynamic_loss_scale[step]

        try:
            if self.args.train.dynamic_batch_enable:
                self.strategy.backward(loss, self.critic, self.critic_optim, name="critic", accumulate=False)
            else:
                self.strategy.backward(
                    loss,
                    self.critic,
                    self.critic_optim,
                    name="critic",
                )
        finally:
            cp_context_stack.close()

        if self.args.train.dynamic_batch_enable:
            if self.replay_buffer.dynamic_optimizer_step[step]:
                self.strategy.optimizer_step(
                    self.critic_optim, self.critic, self.critic_scheduler, name="critic", accumulate=False
                )
        else:
            self.strategy.optimizer_step(self.critic_optim, self.critic, self.critic_scheduler, name="critic")

        # status
        status = {
            "critic_loss": critic_loss.detach().item(),
            "values": masked_mean(values, loss_action_mask).detach().item(),
            "critic_lr": self.critic_scheduler.get_last_lr()[0],
        }
        is_optimizer_step = (
            self.replay_buffer.dynamic_optimizer_step[step]
            if self.args.train.dynamic_batch_enable
            else (step + 1) % self.strategy.accumulated_gradient == 0
        )
        if is_optimizer_step:
            grad_norm = self.strategy.get_grad_norm(self.critic)
            if not self.args.train.dynamic_batch_enable:
                # Log the DeepSpeed-style batch average. Clipping uses the raw
                # norm inside optimizer_step().
                grad_norm /= self.strategy.accumulated_gradient
            status["critic_grad_norm"] = grad_norm
        return status


@ray.remote(num_gpus=1)
class CriticModelActor(BaseModelActor):
    def init_model_from_pretrained(self, strategy: FsdpStrategy, pretrain, max_steps):
        args = strategy.args
        self.disable_ds_ckpt = args.ckpt.disable_ds

        self._setup_distributed(strategy)
        seq_reg_attn = (
            "flash_attention_2" if strategy.args.fsdp.packing_samples else strategy.args.fsdp.attn_implementation
        )
        critic = get_llm_for_sequence_regression(
            pretrain,
            "critic",
            normalize_reward=strategy.args.reward.normalize_enable,
            attn_implementation=seq_reg_attn,
            param_dtype=strategy.args.fsdp.param_dtype,
            lora_rank=strategy.args.fsdp.lora.rank,
            lora_alpha=strategy.args.fsdp.lora.alpha,
            target_modules=strategy.args.fsdp.lora.target_modules,
            lora_dropout=strategy.args.fsdp.lora.dropout,
            device_mesh=strategy.device_mesh,
            moe_mesh=strategy.moe_mesh,
            distributed_config=strategy.distributed_config,
            moe_config=strategy.moe_config,
            activation_checkpointing=args.actor.gradient_checkpointing_enable,
            value_head_prefix=strategy.args.fsdp.value_head_prefix,
            init_value_head=strategy.args.actor.model_name_or_path == strategy.args.critic.model_name_or_path,
            packing_samples=strategy.args.fsdp.packing_samples,
            force_hf_model=strategy.args.fsdp.force_hf_model,
            use_liger_kernel=strategy.args.fsdp.use_liger_kernel,
            moe_aux_loss_coef=args.actor.aux_loss_coef,
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
            strategy.load_ckpt(self.critic, ckpt_path, optimizer=self.critic_optim, scheduler=self.critic_scheduler)

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
                optimizer=self.critic_optim,
                scheduler=self.critic_scheduler,
            )

    @property
    def _sleep_enabled(self) -> bool:
        return bool(getattr(self.strategy.args.fsdp, "enable_sleep", False))

    def reload_states(self):
        """Load critic weights and optimizer state for PPO training."""
        if not self._sleep_enabled:
            return
        if not getattr(self.strategy, "cpu_offload", False):
            self.strategy.move_model_to_device(self.critic, "cuda")
            self.strategy.move_optimizer_to_device(self.critic_optim, "cuda")
        self.critic.train()

    def offload_states(self):
        """Offload critic weights and optimizer state."""
        if not self._sleep_enabled:
            return
        self.strategy.move_optimizer_to_device(self.critic_optim, "cpu")
        self.strategy.move_model_to_device(self.critic, "cpu")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def prepare_for_lp_inference(self):
        """Load critic weights for value inference without optimizer state."""
        if not self._sleep_enabled:
            return
        if not getattr(self.strategy, "cpu_offload", False):
            self.strategy.move_model_to_device(self.critic, "cuda")
        self.critic.eval()

    def offload_after_refit(self):
        """Offload critic weights after value inference."""
        if not self._sleep_enabled:
            return
        self.strategy.move_model_to_device(self.critic, "cpu")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
