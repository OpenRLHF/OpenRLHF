import os
from abc import ABC
from typing import Dict, Tuple

import ray
from tqdm import tqdm

from openrlhf.trainer.ppo_utils.experience_maker import RemoteExperienceMaker
from openrlhf.trainer.ppo_utils.kl_controller import build_kl_controller
from openrlhf.trainer.ppo_utils.replay_buffer import balance_experiences
from openrlhf.trainer.ppo_utils.sample_maker import RemoteSampleGenerator
from openrlhf.trainer.ray.launcher import RayActorGroup
from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.utils.logging_utils import TensorboardLogger, WandbLogger, init_logger
from openrlhf.utils.utils import get_tokenizer

logger = init_logger(__name__)


class BasePPOTrainer(ABC):
    """Training-side base class: model orchestration, logging/eval, PPO steps."""

    def __init__(
        self,
        strategy: DeepspeedStrategy,
        actor_model_group: RayActorGroup,
        critic_model_group: RayActorGroup,
        reward_model_group: RayActorGroup,
        reference_model_group: RayActorGroup,
        vllm_engines,
        tokenizer,
    ) -> None:
        self.strategy = strategy
        self.args = strategy.args

        self.actor_model_group = actor_model_group
        self.critic_model_group = critic_model_group
        self.reward_model_group = reward_model_group
        self.reference_model_group = reference_model_group
        self.vllm_engines = vllm_engines
        self.tokenizer = tokenizer

        self.kl_ctl = build_kl_controller(
            self.args.init_kl_coef,
            self.args.kl_target,
            self.args.kl_horizon,
        )

        self.experience_maker = RemoteExperienceMaker(
            self.actor_model_group,
            self.critic_model_group,
            self.reward_model_group,
            self.reference_model_group,
            self.kl_ctl,
            self.strategy,
            tokenizer,
        )

        # Tracking backends
        self.wandb_logger = WandbLogger(self.args) if self.args.use_wandb else None
        self.tensorboard_logger = TensorboardLogger(self.args) if self.args.use_tensorboard else None

    def fit(self):
        raise NotImplementedError("fit method is not implemented")

    def train_step(self, samples, global_step: int) -> Tuple[Dict, int]:
        # Turn raw samples into PPO-ready trajectories with rewards.
        experiences = self.experience_maker.make_experience_batch(samples)

        # Peek at the first decoded sample for quick sanity check.
        sample0 = [
            self.tokenizer.batch_decode(experiences[0].sequences[0].unsqueeze(0), skip_special_tokens=True)[0],
            experiences[0].info["reward"][0].item(),
        ]
        print(sample0)

        # Balance experiences across DP ranks if needed.
        if self.args.use_dynamic_batch:
            experiences = balance_experiences(experiences, self.args)

        # Push experiences to actor (and critic) shards before PPO.
        refs = self.actor_model_group.async_run_method_batch(method_name="append", experience=experiences)
        if self.critic_model_group is not None:
            refs.extend(self.critic_model_group.async_run_method_batch(method_name="append", experience=experiences))
        ray.get(refs)

        # Perform PPO optimization for actor/critic and gather metrics.
        status = self.ppo_train(global_step)

        # Sync weights to vLLM.
        if self.vllm_engines is not None:
            self.broadcast_to_vllm()

        # Refresh KL controller with the latest measurement.
        if "kl" in status:
            # TODO: KL controller must be FixedKLController; AdaptiveKLController is incompatible here.
            self.kl_ctl.update(status["kl"], self.args.rollout_batch_size * self.args.n_samples_per_prompt)

        status["generated_samples"] = sample0
        return status, global_step + 1

    def ppo_train(self, global_steps: int) -> Dict:
        """Run one PPO train step for critic + actor and return merged status dict."""
        status: dict = {}

        # Decide whether to train critic/actor this round (actor can be frozen initially).
        run_critic = self.critic_model_group is not None
        run_actor = global_steps > self.args.freezing_actor_steps and self.actor_model_group is not None

        def _run_sleep(group, **kwargs):
            # Sleep mode: reload -> fit -> offload (smaller GPU memory).
            ray.get(group.async_run_method(method_name="reload_states"))
            ref = group.async_run_method(method_name="fit", **kwargs)
            status.update(ray.get(ref)[0])
            ray.get(group.async_run_method(method_name="offload_states"))

        if self.args.deepspeed_enable_sleep:
            # Colocated/sleeping: run critic first, then actor.
            if run_critic:
                _run_sleep(self.critic_model_group)
            if run_actor:
                _run_sleep(self.actor_model_group, kl_ctl=self.kl_ctl.value)
        else:
            # Async: start jobs first, then wait and merge results.
            refs = []
            if run_critic:
                refs += self.critic_model_group.async_run_method(method_name="fit")
            if run_actor:
                refs += self.actor_model_group.async_run_method(method_name="fit", kl_ctl=self.kl_ctl.value)

            for result in ray.get(refs):
                status.update(result)

        return status

    def broadcast_to_vllm(self) -> None:
        if self.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "wake_up")

        ray.get(self.actor_model_group.async_run_method(method_name="broadcast_to_vllm"))

        if self.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "sleep")

    def save_logs_and_checkpoints(self, global_step: int, logs_dict=None, client_states=None) -> None:
        logs_dict = logs_dict or {}
        if global_step % self.args.logging_steps == 0:
            if self.wandb_logger:
                self.wandb_logger.log_train(global_step, logs_dict)
            if self.tensorboard_logger:
                self.tensorboard_logger.log_train(global_step, logs_dict)

        # # TODO: Add evaluation mechanism for PPO
        # if global_step % self.args.eval_steps == 0:
        #     self._maybe_run_eval(global_step)

        # save ckpt
        # TODO: save best model on dev, use loss/perplexity/others on whole dev dataset as metric
        client_states = client_states or {}
        if global_step % self.args.save_steps == 0:
            tag = f"global_step{global_step}"
            refs = self.actor_model_group.async_run_method(
                method_name="save_checkpoint", tag=tag, client_states=client_states
            )
            if self.critic_model_group is not None:
                refs.extend(self.critic_model_group.async_run_method(method_name="save_checkpoint", tag=tag))
            ray.get(refs)

    def init_checkpoint_states(self) -> Dict:
        ckpt_path = os.path.join(self.args.ckpt_path, "_actor")
        if self.args.load_checkpoint and os.path.exists(ckpt_path):
            checkpoint_states = ray.get(self.actor_model_group.async_run_method(method_name="get_checkpoint_states"))[
                0
            ]
            logger.info(f"checkpoint_states: {checkpoint_states}")
            return checkpoint_states
        return {"global_step": 0, "episode": 0, "data_loader_state_dict": {}}


@ray.remote
class PPOTrainer(BasePPOTrainer):
    """
    Trainer for Proximal Policy Optimization (PPO) / REINFORCE++ / GRPO / RLOO and their variants.
    Single Controller with Multiple ActorGroups
    """

    def __init__(
        self,
        pretrain: str,
        strategy: DeepspeedStrategy,
        actor_model_group: RayActorGroup,
        critic_model_group: RayActorGroup,
        reward_model_group: RayActorGroup,
        reference_model_group: RayActorGroup,
        vllm_engines,
        prompt_split: str = "train",
        eval_split: str = "test",
        **generate_kwargs,
    ) -> None:
        # Tokenizer is shared across sampler and trainer to avoid duplicated loads.
        tokenizer = get_tokenizer(pretrain, None, "left", strategy, use_fast=not strategy.args.disable_fast_tokenizer)
        # get eval and save steps
        if strategy.args.eval_steps == -1:
            strategy.args.eval_steps = float("inf")  # do not evaluate
        if strategy.args.save_steps == -1:
            strategy.args.save_steps = float("inf")  # do not save ckpt

        # rollout
        self.train_sample_generator = RemoteSampleGenerator(
            strategy=strategy,
            tokenizer=tokenizer,
            vllm_engines=vllm_engines,
            prompt_split=prompt_split,
            generate_kwargs=generate_kwargs,
        )

        # train
        super().__init__(
            strategy,
            actor_model_group,
            critic_model_group,
            reward_model_group,
            reference_model_group,
            vllm_engines,
            tokenizer,
        )

    def fit(self) -> None:
        checkpoint_states = self.init_checkpoint_states()
        # Restore step and start_epoch
        global_step = checkpoint_states["global_step"]
        start_episode = checkpoint_states["episode"]
        # Keep vLLM weights and dataloader states in sync when resuming.
        if global_step:
            self.broadcast_to_vllm()
            self.train_sample_generator.load_state_dict(checkpoint_states["data_loader_state_dict"])

        for episode in range(start_episode, self.args.num_episodes):
            pbar = tqdm(
                range(len(self.train_sample_generator.prompts_dataloader)),
                desc=f"Episode [{episode + 1}/{self.args.num_episodes}]",
                # initial=global_step, # TODO: init step
            )

            while True:
                # Draw one mini-batch of prompts; stop when loader is exhausted.
                samples, pass_rate, prompts_used, is_exhausted = self.train_sample_generator.make_sample_batch()
                if is_exhausted:
                    break

                # Run PPO update on this batch and bump the global step counter.
                status, global_step = self.train_step(samples, global_step)

                # Add generated samples to status dictionary
                if self.args.dynamic_filtering:
                    status["dynamic_filtering_pass_rate"] = pass_rate
                log_statue = {k: v for k, v in status.items() if k not in ["generated_samples"]}
                logger.info(f"✨ Global step {global_step}: {log_statue}")

                # logs/checkpoints
                client_states = {
                    "global_step": global_step,
                    "episode": episode,
                    "data_loader_state_dict": self.train_sample_generator.state_dict(),
                }
                self.save_logs_and_checkpoints(global_step, status, client_states)

                pbar.update(prompts_used)

        # Close trackers
        if self.wandb_logger:
            self.wandb_logger.close()
        if self.tensorboard_logger:
            self.tensorboard_logger.close()

    def get_max_steps(self):
        return self.train_sample_generator.max_steps
