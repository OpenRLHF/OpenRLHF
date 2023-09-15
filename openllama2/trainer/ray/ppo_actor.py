import itertools
import math
import os
from copy import deepcopy
from typing import Dict

import ray
from transformers.trainer import get_scheduler

from openllama2.datasets import PromptDataset, SFTDataset
from openllama2.models import Actor, Critic, RewardModel
from openllama2.trainer import PPOTrainer
from openllama2.trainer.ppo_utils import Experience
from openllama2.utils import DeepspeedStrategy, blending_datasets, get_tokenizer

from .launcher import BasePPORole


class ActorPPOTrainer(PPOTrainer):
    def ppo_train(self):
        # triger remote critic model training
        if self.critic_train_remote:
            critic_status_ref = self.critic.fit.remote()

        status = super().ppo_train()

        # wait remote critic model training done
        if self.critic_train_remote:
            status.update(ray.get(critic_status_ref))
        return status

    def training_step(self, experience: Experience) -> Dict[str, float]:
        return self.training_step_actor(experience)


@ray.remote(num_gpus=1)
class ActorModelRayActor(BasePPORole):
    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain, model_path):
        self._setup_distributed(strategy)
        actor, self.tokenizer = self._from_pretrained(Actor, pretrain, model_path)
        self.prepare_datasets()

        args = strategy.args
        # lora
        if args.lora_rank > 0:
            strategy.print("lora_enable")
            actor.lora_enable(args.lora_rank)

        if args.enable_ema:
            ema_model = deepcopy(actor)
        else:
            ema_model = None

        # configure optimizer
        actor_optim = strategy.create_optimizer(
            actor, lr=args.actor_learning_rate, betas=(0.9, 0.95), weight_decay=args.l2
        )

        # configure scheduler
        num_update_steps_per_episodes = len(self.prompts_dataloader) * args.max_epochs // strategy.accumulated_gradient
        max_steps = math.ceil(args.num_episodes * num_update_steps_per_episodes)
        self.max_steps = max_steps

        actor_scheduler = get_scheduler(
            "cosine",
            actor_optim,
            num_warmup_steps=math.ceil(max_steps * 0.03),
            num_training_steps=max_steps,
        )

        # prepare models/optimizers...
        self.actor, self.actor_optim, self.actor_scheduler = strategy.prepare(
            (actor, actor_optim, actor_scheduler),
            is_rlhf=True,
        )

        if ema_model:
            ema_model.is_ema = True
            self.ema_model = strategy.prepare(ema_model, is_rlhf=True)
            del ema_model.is_ema
        else:
            self.ema_model = None

    def prepare_datasets(self):
        strategy = self.strategy
        args = self.strategy.args

        # prepare datasets
        prompts_data = blending_datasets(
            args.prompt_data,
            args.prompt_data_probs,
            strategy,
            args.seed,
            max_count=100000,
            return_eval=False,
        )
        prompts_data = prompts_data.select(range(min(args.max_samples, len(prompts_data))))
        prompts_dataset = PromptDataset(prompts_data, strategy)
        self.prompts_dataloader = strategy.setup_dataloader(prompts_dataset, args.micro_rollout_batch_size, True, True)

        if args.pretrain_data:
            pretrain_data = blending_datasets(
                args.pretrain_data,
                args.pretrain_data_probs,
                strategy,
                args.seed,
                return_eval=False,
            )
            pretrain_max_len = args.max_len if args.max_len else args.prompt_max_len + args.generate_max_len
            pretrain_dataset = SFTDataset(
                pretrain_data.select(range(min(len(pretrain_data), args.max_epochs * len(prompts_dataset)))),
                self.tokenizer,
                pretrain_max_len,
                strategy,
                pretrain_mode=True,
            )
            self.pretrain_dataloader = itertools.cycle(
                iter(
                    strategy.setup_dataloader(
                        pretrain_dataset,
                        args.micro_train_batch_size,
                        True,
                        True,
                        pretrain_dataset.collate_fn,
                    )
                )
            )
        else:
            self.pretrain_dataloader = None

    def max_steps(self):
        """Return the maximum number of steps."""
        return self.max_steps

    def fit(
        self,
        critic: ray.actor.ActorHandle,
        reward_model: ray.actor.ActorHandle,
        initial_model: ray.actor.ActorHandle,
        critic_train_remote: bool,
    ):
        """Train actor model with prompt datasets."""
        strategy = self.strategy
        args = self.strategy.args

        # configure Trainer
        trainer = ActorPPOTrainer(
            strategy,
            self.actor,
            critic,
            reward_model,
            initial_model,
            ema_model=self.ema_model,
            actor_optim=None,
            critic_optim=None,
            actor_scheduler=self.actor_scheduler,
            critic_scheduler=None,
            max_epochs=args.max_epochs,
            micro_train_batch_size=args.micro_train_batch_size,
            micro_rollout_batch_size=args.micro_rollout_batch_size,
            gradient_checkpointing=args.gradient_checkpointing,
            critic_train_remote=critic_train_remote,
            tokenizer=self.tokenizer,
            prompt_max_len=args.prompt_max_len,
            value_clip=args.value_clip,
            eps_clip=args.eps_clip,
            gamma=args.gamma,
            lambd=args.lambd,
            init_kl_coef=args.init_kl_coef,
            kl_target=args.kl_target,
            ema_beta=0.992,
            ptx_coef=args.ptx_coef,
            max_norm=args.max_norm,
            # fro GPT generation
            do_samples=True,
            max_new_tokens=args.generate_max_len,
            max_length=args.max_len,
            temperature=1,
            top_k=args.top_k,
            top_p=args.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        trainer.fit(
            self.prompts_dataloader,
            self.pretrain_dataloader,
            num_episodes=args.num_episodes,
            rollout_batch_size=args.rollout_batch_size,
        )

    def save_model(self):
        args = self.strategy.args

        # save model checkpoint after fitting on only rank0
        self.strategy.save_model(
            self.ema_model if args.enable_ema else self.actor,
            args.save_path + "/ppo_model.pt",
            only_rank0=True,
        )

        if args.save_hf_model:
            os.makedirs(args.save_path + "/ppo_hf", exist_ok=True)
            self.strategy.save_hf_format(
                self.ema_model if args.enable_ema else self.actor,
                self.tokenizer,
                args.save_path + "/ppo_hf",
            )
