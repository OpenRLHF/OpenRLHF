import asyncio
import os
import time

import ray
from tqdm import tqdm

from openrlhf.trainer.ppo_trainer import BasePPOTrainer
from openrlhf.trainer.ppo_utils import AdaptiveKLController, FixedKLController
from openrlhf.trainer.ppo_utils.experience_maker import RemoteExperienceMaker
from openrlhf.trainer.ray.launcher import RayActorGroup
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

    def fit(self, start_episode, queue, data_loader_state_dict, remote_reward_model=None):
        if data_loader_state_dict:
            self.prompts_dataloader.load_state_dict(data_loader_state_dict)

        for episode in range(start_episode, self.args.num_episodes):
            pbar = tqdm(
                range(self.prompts_dataloader.__len__()),
                desc=f"Generate Episode [{episode + 1}/{self.args.num_episodes}]",
                disable=False,
            )

            filtered_samples = []
            number_of_samples = 0
            queue_log_counter = 0  # Counter for log interval
            for _, rand_prompts, labels in self.prompts_dataloader:
                # Wait until queue is not full
                # To support 1-step off-policy training
                while queue.full():
                    if queue_log_counter % 10 == 0:  # Print log every 10 seconds
                        logger.info("Queue is full, waiting for training to consume samples...")
                    queue_log_counter += 1
                    time.sleep(1)  # Wait for 1 second before checking again

                # Wait for generation to be allowed
                ray.get(self.signal_actor.wait_generating.remote())
                # Block weight updates
                ray.get(self.signal_actor.set_update_weights.remote(False))

                # Generate samples
                # remote_reward_model is used to get rewards for dynamic sampling
                rollout_samples = self.generate_samples(
                    rand_prompts, labels, remote_reward_model=remote_reward_model, **self.generate_kwargs
                )

                # Allow weight updates after generation is done
                ray.get(self.signal_actor.set_update_weights.remote(True))

                pbar.update()

                # dynamic filtering
                pass_rate = None
                if self.args.dynamic_filtering:
                    number_of_samples += len(rollout_samples)
                    # Group individual samples into batches of n_samples size
                    for i in range(0, len(rollout_samples), self.args.n_samples_per_prompt):
                        batch_samples = rollout_samples[i : i + self.args.n_samples_per_prompt]
                        if len(batch_samples) < self.args.n_samples_per_prompt:
                            continue

                        # Calculate average reward for this batch of samples
                        avg_reward = sum(sample.scores[0].item() for sample in batch_samples) / len(batch_samples)

                        # Check if average reward is within the specified range
                        min_reward, max_reward = self.args.dynamic_filtering_reward_range
                        if min_reward + 1e-6 < avg_reward < max_reward - 1e-6:
                            filtered_samples.extend(batch_samples)

                    # Continue sampling if filtered samples are insufficient
                    if len(filtered_samples) / self.args.n_samples_per_prompt < self.args.rollout_batch_size:
                        logger.info(
                            f"filtered_samples {len(filtered_samples) / self.args.n_samples_per_prompt} < rollout_batch_size {self.args.rollout_batch_size}, continue sampling"
                        )
                        continue

                    pass_rate = len(filtered_samples) / number_of_samples * 100
                    logger.info(
                        f"Dynamic filtering pass rate: {pass_rate:.2f}% ({len(filtered_samples)}/{number_of_samples})"
                    )
                    rollout_samples = filtered_samples[: self.args.rollout_batch_size * self.args.n_samples_per_prompt]
                    filtered_samples = []
                    number_of_samples = 0

                queue.put((rollout_samples, episode, self.prompts_dataloader.state_dict(), pass_rate))
        queue.put("done")


@ray.remote
class TrainingActor(BasePPOTrainer):
    def __init__(self, *args, **kwargs):
        self.signal_actor = kwargs.pop("signal_actor")
        self.remote_reward_model = kwargs.pop("remote_reward_model")
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
            self.tokenizer,
            remote_reward_model=self.remote_reward_model,
        )

        self._init_wandb()
        self.eval_dataloader = None

    def _broadcast_to_vllm(self):
        if self.vllm_engines is not None:
            # Block generation
            ray.get(self.signal_actor.set_generating.remote(False))
            # Wait for weight updates to be allowed
            ray.get(self.signal_actor.wait_update_weights.remote())

            # Perform weight update
            ray.get(self.actor_model_group.async_run_method(method_name="broadcast_to_vllm"))

            # Allow generation
            ray.get(self.signal_actor.set_generating.remote(True))

    def fit(self, queue, steps):
        args = self.args

        while True:
            output = queue.get()
            if output == "done":
                break
            rollout_samples, episode, data_loader_state_dict, pass_rate = output

            experiences = self.experience_maker.make_experience_batch(rollout_samples)
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

            # Add generated samples to status dictionary
            if self.args.dynamic_filtering:
                status["dynamic_filtering_pass_rate"] = pass_rate
            logger.info(f"✨ Global step {steps}: {status}")
            status["generated_samples"] = [sample0[0], experiences[0].info["reward"][0]]

            # logs/checkpoints
            client_states = {
                "global_step": steps,
                "episode": episode,
                "data_loader_state_dict": data_loader_state_dict,
            }
            self.save_logs_and_checkpoints(args, steps, None, status, client_states)

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
        actor_model_group: RayActorGroup,
        critic_model_group: RayActorGroup,
        reward_model_group: RayActorGroup,
        reference_model_group: RayActorGroup,
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

        if self.args.remote_rm_url and not self.args.remote_rm_url[0] == "agent":
            from openrlhf.utils.remote_rm_utils import RemoteRewardModel

            self.remote_reward_model = RemoteRewardModel.remote(self.args, self.remote_rm_url)
        else:
            self.remote_reward_model = None

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
        if self.args.eval_steps == -1:
            self.args.eval_steps = float("inf")  # do not evaluate
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
            remote_reward_model=self.remote_reward_model,
            **generate_kwargs,
        )

        from ray.util.queue import Queue

        # the max size is used to control the degree of off-policy
        self.queue = Queue(maxsize=os.environ.get("OPENRLHF_ASYNC_QUEUE_SIZE", 1))

    def fit(self) -> None:
        args = self.args

        # Update initial weights to vLLM engines
        ckpt_path = os.path.join(args.ckpt_path, "_actor")
        if args.load_checkpoint and os.path.exists(ckpt_path):
            checkpoint_states = ray.get(self.actor_model_group.async_run_method(method_name="get_checkpoint_states"))[
                0
            ]
            logger.info(f"checkpoint_states: {checkpoint_states}")
            ray.get(self.trainer_actor._broadcast_to_vllm.remote())
        else:
            checkpoint_states = {"global_step": 0, "episode": 0, "data_loader_state_dict": {}}

        # Restore step and start_epoch
        steps = checkpoint_states["global_step"] + 1
        episode = checkpoint_states["episode"]
        data_loader_state_dict = checkpoint_states["data_loader_state_dict"]

        # Launch async training
        remote_reward_model = self.remote_reward_model if self.args.dynamic_filtering else None
        generator_actor_ref = self.generator_actor.fit.remote(
            episode, self.queue, data_loader_state_dict, remote_reward_model
        )
        trainer_actor_ref = self.trainer_actor.fit.remote(self.queue, steps)
        ray.get([generator_actor_ref, trainer_actor_ref])

    def get_max_steps(self):
        return ray.get(self.generator_actor.get_max_steps.remote())
