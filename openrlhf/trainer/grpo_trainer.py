import math
import os.path
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from openrlhf.models.utils import masked_mean, compute_approx_kl
from openrlhf.utils.distributed_sampler import DistributedSampler

from .ppo_utils import Experience, GRPOExperienceMaker, GRPOReplayBuffer
from . import PPOTrainer

class GRPOTrainer(PPOTrainer):
    def __init__(
        self,
            *args,
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.experience_maker = GRPOExperienceMaker(
            self.actor,
            self.critic,
            self.reward_model,
            self.initial_model,
            self.tokenizer,
            self.prompt_max_len,
            self.kl_ctl,
            self.strategy,
            self.remote_rm_url,
            self.reward_fn,
        )
        self.replay_buffer = GRPOReplayBuffer(
            self.micro_train_batch_size, self.buffer_limit, self.buffer_cpu_offload, self.packing_samples
        )

    def fit(
        self,
        args,
        prompts_dataloader,
        pretrain_dataloader,
        consumed_samples=0,
        num_update_steps_per_episodes=1,
    ) -> None:
        num_rollouts_per_episodes = (
                num_update_steps_per_episodes
                * args.train_batch_size
                // args.max_epochs
                // args.rollout_batch_size
                // args.n_samples_per_prompt
        )

        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = num_rollouts_per_episodes  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        self.prompts_dataloader = prompts_dataloader

        # Restore step and start_epoch
        steps = consumed_samples // args.rollout_batch_size + 1
        start_episode = consumed_samples // args.rollout_batch_size // num_rollouts_per_episodes
        consumed_samples = consumed_samples % (num_rollouts_per_episodes * args.rollout_batch_size)

        for episode in range(start_episode, args.num_episodes):
            if isinstance(self.prompts_dataloader.sampler, DistributedSampler):
                self.prompts_dataloader.sampler.set_epoch(
                    episode, consumed_samples=0 if episode > start_episode else consumed_samples
                )
            pbar = tqdm(
                range(self.prompts_dataloader.__len__()),
                desc=f"Episode [{episode + 1}/{args.num_episodes}]",
                disable=not self.strategy.is_rank_0(),
            )

            for rand_prompts in self.prompts_dataloader:
                rand_prompts = sum([[prompt] * args.n_samples_per_prompt for prompt in rand_prompts], [])
                for i in range(0, len(rand_prompts), args.micro_rollout_batch_size):
                    prompts = rand_prompts[i : i + args.micro_rollout_batch_size]
                    experience = self.experience_maker.make_experience(prompts, **self.generate_kwargs)
                    if i == 0:
                        output = self.tokenizer.batch_decode(
                            experience.sequences[0].unsqueeze(0), skip_special_tokens=True
                        )
                        self.strategy.print(output)
                    self.replay_buffer.append(experience)
                self.replay_buffer.calculate_advantages()
                self.replay_buffer.dic2items()
                torch.cuda.empty_cache()
                self.replay_buffer.normalize("advantages", self.strategy)
                status = self.grpo_train()
                self.replay_buffer.clear()
                torch.cuda.empty_cache()

                if "kl" in status:
                    self.kl_ctl.update(status["kl"], args.rollout_batch_size)
                pbar.set_postfix(status)

                # logs/checkpoints
                client_states = {"consumed_samples": steps * args.rollout_batch_size}
                self.save_logs_and_checkpoints(args, steps, pbar, status, client_states)

                pbar.update()
                steps = steps + 1

    def grpo_train(self):
        # replay buffer may be empty at first, we should rebuild at each training
        dataloader = DataLoader(
            self.replay_buffer,
            batch_size=self.replay_buffer.sample_batch_size,
            shuffle=False,
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
                # weighted mean for kl
                if "kl" in status:
                    status["kl"] *= status["response_length"]
                    status = self.strategy.all_reduce(status)
                    status["kl"] /= status["response_length"]

                short_status = {}

                if "policy_loss" in status:
                    short_status = {
                        "pg": status["policy_loss"],
                        "rm": status["reward"],
                        "glen": status["response_length"],
                        "tlen": status["total_length"],
                        "kl": status["kl"],
                        "act_lr": status["actor_lr"],
                    }

                status_list.append(status)
                pbar.set_postfix(short_status)

        if status_list:
            status_mean = status_list[0]
            for m in status_list[1:]:
                for k, v in m.items():
                    status_mean[k] += v
            for k in status_mean.keys():
                status_mean[k] /= len(status_list)
        return status_mean

    def training_step(self, experience: Experience, global_steps = None) -> Dict[str, float]:
        self.actor.train()

        num_actions = experience.action_mask.size(1)
        # actor loss
        action_log_probs, output = self.actor(
            experience.sequences, num_actions, attention_mask=experience.attention_mask, return_output=True
        )
        # loss function
        actor_loss = self.actor_loss_fn(
            action_log_probs,
            experience.action_log_probs,
            experience.advantages,
            action_mask=experience.action_mask,
        )
        kl = compute_approx_kl(action_log_probs,
                               experience.action_log_probs,
                               action_mask=experience.action_mask,
                               with_low_variance=True)
        kl = masked_mean(kl, experience.action_mask, dim=-1)
        # mixtral
        if self.aux_loss:
            aux_loss = output.aux_loss
        else:
            aux_loss = 0
        loss = actor_loss + self.kl_ctl.value * kl.mean() + aux_loss * self.args.aux_loss_coef
        self.strategy.backward(loss, self.actor, self.actor_optim)

        self.strategy.optimizer_step(self.actor_optim, self.actor, self.actor_scheduler, name="actor")
        if self.ema_model:
            self.strategy.moving_average(self.actor, self.ema_model, self.ema_beta, "cpu")

        # status
        status = {"policy_loss": actor_loss.item(), "actor_lr": self.actor_scheduler.get_last_lr()[0]}
        status["kl"] = (
                (kl.to("cpu") * experience.info["response_length"]).sum() / experience.info["response_length"].sum()
        ).item()
        for k, v in experience.info.items():
            status[k] = v.mean().item()
        return status

    def _save_checkpoint(self, args, tag, client_states):
        self.strategy.save_ckpt(
            self.actor.model,
            os.path.join(args.ckpt_path, "_actor"),
            tag,
            args.max_ckpt_num,
            args.max_ckpt_mem,
            client_states,
        )
