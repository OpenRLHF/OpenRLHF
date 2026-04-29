import gc
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
from openrlhf.utils.fsdp import FsdpStrategy

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
            dynamic_batch=self.args.train.dynamic_batch_enable,
        )

        self.critic_loss_fn = ValueLoss(value_clip)

        # Mixtral 8x7b
        self.aux_loss = self.args.actor.aux_loss_coef > 1e-8

    def ppo_train(self):
        # replay buffer may be empty at first, we should rebuild at each training
        if self.args.train.dynamic_batch_enable:
            self.replay_buffer.setup_dynamic_batch(self.strategy)

        should_shuffle = self.args.fsdp.tp_size <= 1 and not self.args.train.dynamic_batch_enable
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
            accum_window = []
            accum_steps = self.strategy.accumulated_gradient
            for step, experience in enumerate(pbar):
                experience.to_device(device)
                if self.args.train.dynamic_batch_enable:
                    status = self.training_step(experience, step)
                    status = self.strategy.all_reduce(status)
                    status_list.append(status)
                    pbar.set_postfix(status)
                    continue

                accum_window.append((step, experience))
                if len(accum_window) < accum_steps:
                    continue

                local_num_tokens = torch.zeros((), dtype=torch.float32, device=torch.device("cuda", device))
                local_batch_size = torch.zeros((), dtype=torch.float32, device=torch.device("cuda", device))
                for _, window_experience in accum_window:
                    local_num_tokens += window_experience.action_mask.sum()
                    local_batch_size += (window_experience.action_mask.sum(dim=-1) > 0).sum()
                global_num_tokens = self.strategy.global_token_count(local_num_tokens)
                global_batch_size = self.strategy.global_token_count(local_batch_size)

                for window_step, window_experience in accum_window:
                    status = self.training_step(
                        window_experience,
                        window_step,
                        global_num_tokens=global_num_tokens,
                        global_batch_size=global_batch_size,
                        loss_dp_size=self.strategy.dp_size,
                        loss_report_scale=1,
                        scale_loss_by_accumulation=False,
                    )
                    status = self.strategy.all_reduce(status)
                    status_list.append(status)
                    pbar.set_postfix(status)
                accum_window = []

            # Trailing partial windows are dropped on purpose — see ppo_actor.py
            # for the rationale (would advance time_steps but never fire
            # optimizer_step, leaking grads into the next epoch).
            if accum_window:
                self.strategy.print(
                    f"[Critic] dropping {len(accum_window)} trailing microbatches "
                    f"(< accum_steps={accum_steps}); check that train.batch_size, "
                    f"micro_batch_size, dp_size, and rollout n_samples align."
                )

        if status_list:
            status_mean = status_list[0]
            for m in status_list[1:]:
                for k, v in m.items():
                    status_mean[k] += v
            for k in status_mean.keys():
                status_mean[k] /= len(status_list)
        return status_mean

    def training_step(
        self,
        experience: Experience,
        step: int,
        global_num_tokens=None,
        global_batch_size=None,
        loss_dp_size=None,
        loss_report_scale: int = 1,
        scale_loss_by_accumulation: bool = True,
    ) -> Dict[str, float]:
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
            values_allgather=True,
            packed_seq_lens=packed_seq_lens,
        )

        if global_num_tokens is None:
            global_num_tokens = self.strategy.global_token_count(experience.action_mask)
        if global_batch_size is None:
            global_batch_size = self.strategy.global_token_count((experience.action_mask.sum(dim=-1) > 0).sum())
        if loss_dp_size is None:
            loss_dp_size = self.strategy.dp_size

        # loss function
        critic_loss = self.critic_loss_fn(
            values,
            old_values,
            returns,
            action_mask=experience.action_mask,
            dp_size=loss_dp_size,
            batch_num_tokens=global_num_tokens,
            global_batch_size=global_batch_size,
        )
        # mixtral
        if self.aux_loss:
            aux_loss = getattr(output, "aux_loss", 0)
        else:
            aux_loss = 0
        aux_scale = 1 if scale_loss_by_accumulation else self.strategy.accumulated_gradient
        loss = critic_loss + aux_loss * self.args.actor.aux_loss_coef / aux_scale
        if self.args.train.dynamic_batch_enable:
            loss = loss * self.replay_buffer.dynamic_loss_scale[step]

        if self.args.train.dynamic_batch_enable:
            self.strategy.backward(loss, self.critic, self.critic_optim, name="critic", accumulate=False)
            if self.replay_buffer.dynamic_optimizer_step[step]:
                self.strategy.optimizer_step(
                    self.critic_optim, self.critic, self.critic_scheduler, name="critic", accumulate=False
                )
        else:
            self.strategy.backward(
                loss,
                self.critic,
                self.critic_optim,
                name="critic",
                scale_loss_by_accumulation=scale_loss_by_accumulation,
            )
            self.strategy.optimizer_step(self.critic_optim, self.critic, self.critic_scheduler, name="critic")

        # status
        status = {
            "critic_loss": (critic_loss.detach() / loss_report_scale).item(),
            "values": masked_mean(values, experience.action_mask).detach().item(),
            "critic_lr": self.critic_scheduler.get_last_lr()[0],
            "critic_grad_norm": self.strategy.get_grad_norm(self.critic),
        }
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
            load_in_4bit=strategy.args.fsdp.load_in_4bit,
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
            use_liger_kernel=strategy.args.fsdp.use_liger_kernel,
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
        """Hybrid/sleep entry: bring critic + optimizer state back to GPU."""
        if not self._sleep_enabled:
            return
        self.strategy.move_model_to_device(self.critic, "cuda")
        self.strategy.move_optimizer_to_device(self.critic_optim, "cuda")
        self.critic.train()

    def offload_states(self):
        """Hybrid/sleep exit: ship critic state off-GPU."""
        if not self._sleep_enabled:
            return
        self.strategy.move_optimizer_to_device(self.critic_optim, "cpu")
        self.strategy.move_model_to_device(self.critic, "cpu")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def prepare_for_lp_inference(self):
        """Value forward only needs critic model on GPU; optimizer stays cpu
        to leave the device free for the inference pass."""
        if not self._sleep_enabled:
            return
        self.strategy.move_model_to_device(self.critic, "cuda")
        self.critic.eval()

    def offload_after_refit(self):
        """Tear down after a value-forward (model→cpu only)."""
        if not self._sleep_enabled:
            return
        self.strategy.move_model_to_device(self.critic, "cpu")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
