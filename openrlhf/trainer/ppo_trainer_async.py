import asyncio
import os

import ray
from tqdm import tqdm

from openrlhf.trainer.ppo_trainer import BasePPOTrainer
from openrlhf.trainer.ppo_utils import AdaptiveKLController, FixedKLController
from openrlhf.trainer.ppo_utils.experience_maker import RemoteExperienceMaker
from openrlhf.trainer.ray.launcher import PPORayActorGroup
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)


@ray.remote(num_cpus=0)
class SignalActor:
    def __init__(self):
        self.generating_event = asyncio.Event()
        self.update_weights_event = asyncio.Event()
        self.generating_event.set()  # Initially allow generation
        self.update_weights_event.set()  # Initially allow weight updates

    async def wait_generating(self):
        """Wait for generation to be allowed."""
        return await self.generating_event.wait()

    async def wait_update_weights(self):
        """Wait for weight update to be allowed."""
        return await self.update_weights_event.wait()

    def set_generating(self, allow_generating):
        """Set generation state.

        Args:
            is_generating: True to allow generation, False to block it
        """
        if allow_generating:
            self.generating_event.set()
        else:
            self.generating_event.clear()

    def set_update_weights(self, allow_updating):
        """Set weight update state.

        Args:
            is_updating: True to allow weight updates, False to block it
        """
        if allow_updating:
            self.update_weights_event.set()
        else:
            self.update_weights_event.clear()


@ray.remote
class GenerateSamplesActor(BasePPOTrainer):
    def __init__(self, *args, **kwargs):
        self.signal_actor = kwargs.pop("signal_actor")
        super().__init__(*args, **kwargs)

        self.samples_generator = self.generator_cls(
            self.vllm_engines,
            self.strategy,
            self.tokenizer,
            self.prompt_max_len,
        )
        self.prepare_datasets()

    def generate_samples(self, prompts, labels, **generate_kwargs):
        return self.samples_generator.generate_samples(prompts, labels, **generate_kwargs)

    def fit(self, start_episode, consumed_samples, queue):
        for episode in range(start_episode, self.args.num_episodes):
            self.prompts_dataloader.sampler.set_epoch(
                episode, consumed_samples=0 if episode > start_episode else consumed_samples
            )
            pbar = tqdm(
                range(self.prompts_dataloader.__len__()),
                desc=f"Generate Episode [{episode + 1}/{self.args.num_episodes}]",
                disable=False,
            )

            for _, rand_prompts, labels in self.prompts_dataloader:
                # Wait for generation to be allowed
                ray.get(self.signal_actor.wait_generating.remote())
                # Block weight updates
                ray.get(self.signal_actor.set_update_weights.remote(False))

                # Generate samples
                rollout_samples = self.generate_samples(rand_prompts, labels, **self.generate_kwargs)

                # Allow weight updates after generation is done
                ray.get(self.signal_actor.set_update_weights.remote(True))

                queue.put(rollout_samples)
                pbar.update()
        queue.put("done")

    def get_prompts_dataloader_len(self):
        return self.prompts_dataloader.__len__()


@ray.remote
class TrainingActor(BasePPOTrainer):
    def __init__(self, *args, **kwargs):
        self.signal_actor = kwargs.pop("signal_actor")
        super().__init__(*args, **kwargs)

        if self.kl_target:
            self.kl_ctl = AdaptiveKLController(self.init_kl_coef, self.kl_target, self.kl_horizon)
        else:
            self.kl_ctl = FixedKLController(self.init_kl_coef)

        self.experience_maker = RemoteExperienceMaker(
            self.actor_model_group,
            self.critic_model_group,
            self.reward_model_group,
            self.reference_model_group,
            self.kl_ctl,
            self.strategy,
            self.remote_rm_url,
        )

        self._init_wandb()
        self.eval_dataloader = None

    def _update_weights_to_vllm(self):
        if self.vllm_engines is not None:
            # Block generation
            ray.get(self.signal_actor.set_generating.remote(False))
            # Wait for weight updates to be allowed
            ray.get(self.signal_actor.wait_update_weights.remote())

            # Perform weight update
            ray.get(self.actor_model_group.async_run_method(method_name="broadcast_to_vllm"))

            # Allow generation
            ray.get(self.signal_actor.set_generating.remote(True))

    def fit(self, queue, steps, pbar_steps):
        args = self.args
        pbar = tqdm(
            range(pbar_steps),
            desc=f"Training Process",
            disable=False,
        )

        while True:
            rollout_samples = queue.get()
            if rollout_samples == "done":
                break

            experiences = self.experience_maker.make_experience_list(rollout_samples)
            sample0 = self.tokenizer.batch_decode(experiences[0].sequences[0].unsqueeze(0), skip_special_tokens=True)
            print(sample0)
            refs = self.actor_model_group.async_run_method_batch(method_name="append", experience=experiences)
            if self.critic_model_group is not None:
                refs.extend(
                    self.critic_model_group.async_run_method_batch(method_name="append", experience=experiences)
                )
            ray.get(refs)

            status = self.ppo_train(steps)

            if "kl" in status:
                self.kl_ctl.update(status["kl"], args.rollout_batch_size * args.n_samples_per_prompt)
            pbar.set_postfix(status)

            # Add generated samples to status dictionary
            status["generated_samples"] = [sample0[0], experiences[0].info["reward"][0]]
            # logs/checkpoints
            client_states = {"consumed_samples": steps * args.rollout_batch_size}
            self.save_logs_and_checkpoints(args, steps, pbar, status, client_states)

            pbar.update()
            steps = steps + 1

        if self._wandb is not None:
            self._wandb.finish()
        if self._tensorboard is not None:
            self._tensorboard.close()


@ray.remote
class PPOTrainerAsync:
    def __init__(
        self,
        pretrain: str,
        strategy: DeepspeedStrategy,
        actor_model_group: PPORayActorGroup,
        critic_model_group: PPORayActorGroup,
        reward_model_group: PPORayActorGroup,
        reference_model_group: PPORayActorGroup,
        vllm_engines=None,
        prompt_max_len: int = 120,
        dataloader_pin_memory: bool = True,
        prompt_split: str = "train",
        eval_split: str = "test",
        **generate_kwargs,
    ) -> None:
        super().__init__()

        self.args = strategy.args
        self.strategy = strategy
        self.actor_model_group = actor_model_group
        self.critic_model_group = critic_model_group
        self.reward_model_group = reward_model_group
        self.reference_model_group = reference_model_group
        self.vllm_engines = vllm_engines
        self.prompt_max_len = prompt_max_len

        # Create signal actor for synchronization
        self.signal_actor = SignalActor.remote()

        self.generator_actor = GenerateSamplesActor.remote(
            pretrain,
            strategy,
            actor_model_group,
            critic_model_group,
            reward_model_group,
            reference_model_group,
            vllm_engines,
            prompt_max_len,
            dataloader_pin_memory,
            prompt_split,
            eval_split,
            signal_actor=self.signal_actor,
            **generate_kwargs,
        )

        # get eval and save steps
        self.num_rollouts_per_episodes = ray.get(self.generator_actor.get_prompts_dataloader_len.remote())
        if self.args.eval_steps == -1:
            self.args.eval_steps = self.num_rollouts_per_episodes  # Evaluate once per epoch
        if self.args.save_steps == -1:
            self.args.save_steps = float("inf")  # do not save ckpt

        self.trainer_actor = TrainingActor.remote(
            pretrain,
            strategy,
            actor_model_group,
            critic_model_group,
            reward_model_group,
            reference_model_group,
            vllm_engines,
            prompt_max_len,
            dataloader_pin_memory,
            prompt_split,
            eval_split,
            signal_actor=self.signal_actor,
            **generate_kwargs,
        )

        from ray.util.queue import Queue

        # the max size is used to control the degree of off-policy
        self.queue = Queue(maxsize=os.environ.get("OPENRLHF_ASYNC_QUEUE_SIZE", 3))

    def fit(self) -> None:
        args = self.args

        # Update initial weights to vLLM engines
        ckpt_path = os.path.join(args.ckpt_path, "_actor")
        if args.load_checkpoint and os.path.exists(ckpt_path) and not self.vllm_engines is None:
            ray.get(self.trainer_actor._update_weights_to_vllm.remote())

        # Restore step and start_epoch
        consumed_samples = ray.get(self.actor_model_group.async_run_method(method_name="get_consumed_samples"))[0]
        steps = consumed_samples // args.rollout_batch_size + 1
        start_episode = consumed_samples // args.rollout_batch_size // self.num_rollouts_per_episodes
        consumed_samples = consumed_samples % (self.num_rollouts_per_episodes * args.rollout_batch_size)
        pbar_steps = self.num_rollouts_per_episodes * args.num_episodes - steps

        # Launch async training
        generator_actor_ref = self.generator_actor.fit.remote(start_episode, consumed_samples, self.queue)
        trainer_actor_ref = self.trainer_actor.fit.remote(self.queue, steps, pbar_steps)
        ray.get([generator_actor_ref, trainer_actor_ref])

    def get_max_steps(self):
        return self.generator_actor.get_max_steps.remote()
