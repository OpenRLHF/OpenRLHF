import os
from abc import ABC

import ray
from tqdm import tqdm

from openrlhf.trainer.ppo_utils.experience_maker import RemoteExperienceMaker
from openrlhf.trainer.ppo_utils.kl_controller import build_kl_controller
from openrlhf.trainer.ppo_utils.misc import (
    ensure_remote_rm,
    init_tensorboard,
    init_wandb,
    normalize_interval_config,
)
from openrlhf.trainer.ppo_utils.replay_buffer import balance_experiences
from openrlhf.trainer.ppo_utils.sample_maker import RemoteSampleStreamer
from openrlhf.trainer.ray.launcher import RayActorGroup
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.utils.logging_utils import init_logger
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
        prompt_max_len: int = 120,
        **generate_kwargs,
    ) -> None:
        self.strategy = strategy
        self.args = strategy.args

        self.actor_model_group = actor_model_group
        self.critic_model_group = critic_model_group
        self.reward_model_group = reward_model_group
        self.reference_model_group = reference_model_group
        self.vllm_engines = vllm_engines
        self.tokenizer = tokenizer

        self.prompt_max_len = prompt_max_len
        self.freezing_actor_steps = getattr(self.args, "freezing_actor_steps", -1)

        self.remote_reward_model = ensure_remote_rm(
            self.args, self.args.remote_rm_url, generate_kwargs.pop("remote_reward_model", None)
        )
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
            remote_reward_model=self.remote_reward_model,
        )

        # Tracking backends
        self._wandb, self._wandb_samples_table = init_wandb(self.args)
        self._tensorboard = init_tensorboard(self.args)

        self._eval_runner = None
        self.generate_kwargs = generate_kwargs
        normalize_interval_config(self.args)

    def ppo_train(self, global_steps):
        """Run one PPO train step for critic + actor and return merged status dict."""
        status: dict = {}

        run_critic = self.critic_model_group is not None
        run_actor = global_steps > self.freezing_actor_steps and self.actor_model_group is not None

        def _run_sleep(group, **kwargs):
            # Sleep mode: reload -> fit -> offload (smaller GPU memory).
            ray.get(group.async_run_method(method_name="reload_states"))
            ref = group.async_run_method(method_name="fit", **kwargs)
            status.update(ray.get(ref)[0])
            ray.get(group.async_run_method(method_name="offload_states"))

        if self.strategy.args.deepspeed_enable_sleep:
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

    def train_step(self, experiences, global_step):
        # Peek at the first decoded sample for quick sanity check.
        sample_text = self.tokenizer.batch_decode(experiences[0].sequences[0].unsqueeze(0), skip_special_tokens=True)[
            0
        ]
        sample_reward = experiences[0].info["reward"][0]
        print(sample_text, sample_reward)

        # Balance experiences across DP ranks if needed.
        if self.args.use_dynamic_batch:
            experiences = balance_experiences(experiences, self.args)

        # Push experiences to actor (and critic) shards before PPO.
        refs = self.actor_model_group.async_run_method_batch(method_name="append", experience=experiences)
        if self.critic_model_group is not None:
            refs.extend(self.critic_model_group.async_run_method_batch(method_name="append", experience=experiences))
        ray.get(refs)

        status = self.ppo_train(global_step)

        # Sync weights to vLLM.
        if self.vllm_engines is not None:
            self._broadcast_to_vllm()

        # Refresh KL controller with the latest measurement.
        if "kl" in status:
            self.kl_ctl.update(status["kl"], self.args.rollout_batch_size * self.args.n_samples_per_prompt)

        status["generated_samples"] = [sample_text, sample_reward]
        return global_step + 1, status

    def _broadcast_to_vllm(self):
        if self.strategy.args.vllm_enable_sleep:
            from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call

            batch_vllm_engine_call(self.vllm_engines, "wake_up")

        ray.get(self.actor_model_group.async_run_method(method_name="broadcast_to_vllm"))

        if self.strategy.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "sleep")

    def save_logs_and_checkpoints(self, global_step, logs_dict=None, client_states=None):
        logs_dict = logs_dict or {}
        if global_step % self.args.logging_steps == 0:
            if self._wandb is not None:
                # Add generated samples to wandb using Table
                if logs_dict.get("generated_samples"):
                    # https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
                    new_table = self._wandb.Table(
                        columns=self._wandb_samples_table.columns, data=self._wandb_samples_table.data
                    )
                    new_table.add_data(global_step, *logs_dict.pop("generated_samples"))
                    self._wandb_samples_table = new_table
                    self._wandb.log({"train/generated_samples": new_table})

                metrics = {k: v for k, v in logs_dict.items() if v is not None}
                logs = {"train/%s" % k: v for k, v in {**metrics, "global_step": global_step}.items()}
                self._wandb.log(logs)
            elif self._tensorboard is not None:
                for k, v in logs_dict.items():
                    if k == "generated_samples" and v is not None:
                        # Record generated samples in TensorBoard using simple text format
                        text, reward = v
                        formatted_text = f"Sample:\n{text}\n\nReward: {reward:.4f}"
                        self._tensorboard.add_text("train/generated_samples", formatted_text, global_step)
                    elif v is not None:
                        self._tensorboard.add_scalar(f"train/{k}", v, global_step)

        # TODO: Add evaluation mechanism for PPO
        if global_step % self.args.eval_steps == 0:
            self._maybe_run_eval(global_step)

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

    def _log_eval_metrics(self, global_step, logs):
        if not logs:
            return
        if self._wandb is not None:
            logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": global_step}.items()}
            self._wandb.log(logs)
        elif self._tensorboard is not None:
            for k, v in logs.items():
                self._tensorboard.add_scalar(f"eval/{k}", v, global_step)

    def _build_eval_runner(self, evaluator, *, before_eval=None, after_eval=None):
        """Wrap evaluator into a callable so subclasses only decide when to run."""
        if evaluator is None:
            return None

        def _run(global_step):
            if before_eval:
                before_eval()
            try:
                return evaluator.run(
                    global_step,
                    self.args.eval_temperature,
                    self.args.eval_n_samples_per_prompt,
                )
            finally:
                if after_eval:
                    after_eval()

        return _run

    def _load_checkpoint_states(self):
        ckpt_path = os.path.join(self.args.ckpt_path, "_actor")
        if self.args.load_checkpoint and os.path.exists(ckpt_path):
            checkpoint_states = ray.get(self.actor_model_group.async_run_method(method_name="get_checkpoint_states"))[
                0
            ]
            logger.info(f"checkpoint_states: {checkpoint_states}")
            return checkpoint_states
        return {"global_step": 0, "episode": 0, "data_loader_state_dict": {}}

    def _maybe_run_eval(self, global_step):
        if self._eval_runner is None:
            return
        logs = self._eval_runner(global_step)
        self._log_eval_metrics(global_step, logs)


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
        vllm_engines=None,
        prompt_max_len: int = 120,
        train_split: str = "train",
        eval_split: str = "test",
        **generate_kwargs,
    ) -> None:
        tokenizer = get_tokenizer(pretrain, None, "left", strategy, use_fast=not strategy.args.disable_fast_tokenizer)

        super().__init__(
            strategy,
            tokenizer,
            actor_model_group,
            critic_model_group,
            reward_model_group,
            reference_model_group,
            vllm_engines,
            prompt_max_len,
            **generate_kwargs,
        )

        self.train_streamer = RemoteSampleStreamer(
            strategy,
            tokenizer,
            vllm_engines,
            prompt_max_len,
            train_split,
            generate_kwargs,
        )

    def fit(self) -> None:
        checkpoint_states = self._load_checkpoint_states()
        # Restore step and start_epoch
        global_step = checkpoint_states["global_step"]
        start_episode = checkpoint_states["episode"]
        # Keep vLLM weights and dataloader states in sync when resuming.
        if global_step:
            ray.get(self.trainer_actor._broadcast_to_vllm.remote())
            self.train_streamer.load_state_dict(checkpoint_states["data_loader_state_dict"])

        for episode in range(start_episode, self.args.num_episodes):
            pbar = tqdm(
                range(len(self.train_streamer.prompts_dataloader)),
                desc=f"Episode [{episode + 1}/{self.args.num_episodes}]",
                # initial=steps, # TODO
            )

            while True:
                samples, pass_rate, prompts_used, is_exhausted = self.train_streamer.make_sample_batch()
                if is_exhausted:
                    break

                experiences = self.experience_maker.make_experience_batch(samples)
                global_step, status = self.train_step(experiences, global_step)

                if self.args.dynamic_filtering:
                    status["dynamic_filtering_pass_rate"] = pass_rate
                logger.info(f"✨ Global step {global_step}: {status}")

                client_states = {
                    "global_step": global_step,
                    "episode": episode,
                    "data_loader_state_dict": self.train_streamer.state_dict(),
                }
                self.save_logs_and_checkpoints(global_step, status, client_states)

                pbar.update(prompts_used)

        # close trackers
        if self._wandb is not None:
            self._wandb.finish()
        if self._tensorboard is not None:
            self._tensorboard.close()

    def get_max_steps(self):
        return self.train_streamer.max_steps
