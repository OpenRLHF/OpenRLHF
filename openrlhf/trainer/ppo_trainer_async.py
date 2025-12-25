import asyncio

import ray
from ray.util.queue import Queue
from tqdm import tqdm

from openrlhf.trainer.ppo_trainer import BasePPOTrainer
from openrlhf.trainer.ppo_utils.experience_maker import RolloutSampler
from openrlhf.trainer.ray.launcher import RayActorGroup
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.utils import get_tokenizer

logger = init_logger(__name__)


@ray.remote(num_cpus=0)
class GenerationUpdateLock:
    """Mutual exclusion gate between generation and weight updates."""

    def __init__(self):
        self._cond = asyncio.Condition()
        self._generating = False
        self._updating = False

    async def acquire_for_generation(self):
        """Block until no update is active, then mark generation as active."""
        async with self._cond:
            while self._updating:
                await self._cond.wait()
            self._generating = True

    async def release_for_generation(self):
        """Mark generation as inactive and wake any waiting updates."""
        async with self._cond:
            self._generating = False
            self._cond.notify_all()

    async def acquire_for_update(self):
        """Block until no generation is active, then mark update as active."""
        async with self._cond:
            while self._generating:
                await self._cond.wait()
            self._updating = True

    async def release_for_update(self):
        """Mark update as inactive and wake any waiting generators."""
        async with self._cond:
            self._updating = False
            self._cond.notify_all()


@ray.remote
class GenerateSamplesActor:
    def __init__(
        self,
        pretrain,
        strategy,
        vllm_engines,
        prompt_split="train",
        eval_split="test",
        *,
        generation_update_lock,
        rollout_queue,
        rollout_slots,
        **generate_kwargs,
    ):
        tokenizer = get_tokenizer(pretrain, None, "left", strategy, use_fast=not strategy.args.disable_fast_tokenizer)

        self.strategy = strategy
        self.args = strategy.args

        self.rollout_sampler = RolloutSampler(
            strategy=strategy,
            tokenizer=tokenizer,
            vllm_engines=vllm_engines,
            prompt_split=prompt_split,
            generate_kwargs=generate_kwargs,
        )

        self.generation_update_lock = generation_update_lock
        self.rollout_queue = rollout_queue
        # Acts like a counting semaphore for rollout_queue capacity (cross-actor safe).
        self.rollout_slots = rollout_slots

    def get_max_steps(self):
        return self.rollout_sampler.max_steps

    def load_state_dict(self, state_dict):
        self.rollout_sampler.load_state_dict(state_dict)

    def fit(self, episode, total_consumed_prompts):
        for episode in range(episode, self.args.num_episodes):
            dataset_length = len(self.rollout_sampler.prompts_dataloader)
            pbar = tqdm(
                range(dataset_length),
                desc=f"Episode [{episode + 1}/{self.args.num_episodes}]",
                initial=total_consumed_prompts % dataset_length,
            )
            while True:
                # Backpressure: acquire a slot before generating.
                self.rollout_slots.get(block=True)

                # Pause updates while we sample to keep outputs consistent.
                ray.get(self.generation_update_lock.acquire_for_generation.remote())
                try:
                    # Draw one mini-batch of prompts; stop when loader is exhausted.
                    rollout_samples, filter_pass_rate, prompts_consumed, is_exhausted = (
                        self.rollout_sampler.sample_batch()
                    )
                    total_consumed_prompts += prompts_consumed
                finally:
                    ray.get(self.generation_update_lock.release_for_generation.remote())
                if is_exhausted:
                    break

                # Send the produced batch to the controller for training.
                client_states = {
                    "episode": episode,
                    "total_consumed_prompts": total_consumed_prompts,
                    "data_loader_state_dict": self.rollout_sampler.state_dict(),
                }
                self.rollout_queue.put((rollout_samples, client_states, filter_pass_rate), block=True)
                pbar.update(prompts_consumed)

        self.rollout_queue.put("done", block=True)


@ray.remote
class TrainingActor(BasePPOTrainer):
    def __init__(
        self,
        pretrain,
        strategy,
        actor_model_group,
        critic_model_group,
        reward_model_group,
        reference_model_group,
        vllm_engines,
        generation_update_lock,
        rollout_queue,
        rollout_slots,
    ):
        tokenizer = get_tokenizer(pretrain, None, "left", strategy, use_fast=not strategy.args.disable_fast_tokenizer)

        super().__init__(
            strategy,
            actor_model_group,
            critic_model_group,
            reward_model_group,
            reference_model_group,
            vllm_engines,
            tokenizer,
        )
        self.generation_update_lock = generation_update_lock
        self.rollout_queue = rollout_queue
        # Acts like a counting semaphore for rollout_queue capacity (cross-actor safe).
        self.rollout_slots = rollout_slots

    def fit(self, global_step):
        while True:
            # Draw one mini-batch of prompts; stop when loader is exhausted.
            payload = self.rollout_queue.get(block=True)
            if payload == "done":
                break
            rollout_samples, client_states, filter_pass_rate = payload

            # Release a slot after batch is consumed to unblock generator.
            self.rollout_slots.put(None, block=True)

            # Run PPO update on this batch and bump the global step counter.
            status, global_step = self.train_step(rollout_samples, global_step)

            # Add generated samples to status dictionary
            if self.args.dynamic_filtering:
                status["dynamic_filtering_pass_rate"] = filter_pass_rate
            log_status = {k: v for k, v in status.items() if k not in ["generated_samples"]}
            logger.info(f"✨ Global step {global_step}: {log_status}")

            # logs/checkpoints
            client_states.update({"global_step": global_step})
            self.save_logs_and_checkpoints(global_step, status, client_states)

        # Close trackers
        if self.wandb_logger:
            self.wandb_logger.close()
        if self.tensorboard_logger:
            self.tensorboard_logger.close()

    def broadcast_to_vllm(self):
        ray.get(self.generation_update_lock.acquire_for_update.remote())
        try:
            # broadcast
            super().broadcast_to_vllm()
        finally:
            ray.get(self.generation_update_lock.release_for_update.remote())


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
        vllm_engines,
        prompt_split: str = "train",
        eval_split: str = "test",
        **generate_kwargs,
    ) -> None:
        args = strategy.args
        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = float("inf")  # do not evaluate
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        queue_size = getattr(args, "async_queue_size", 1)
        if queue_size <= 0:
            raise ValueError(f"async_queue_size must be positive, got {queue_size}")
        logger.info(f"queue_size={queue_size}")

        # Token pool used as a counting semaphore for rollout_queue capacity.
        self.rollout_slots = Queue(maxsize=queue_size)
        for _ in range(queue_size):
            self.rollout_slots.put(None, block=True)

        self.rollout_queue = Queue(maxsize=queue_size)
        self.generation_update_lock = GenerationUpdateLock.remote()

        # rollout
        self.generator_actor = GenerateSamplesActor.remote(
            pretrain=pretrain,
            strategy=strategy,
            vllm_engines=vllm_engines,
            prompt_split=prompt_split,
            eval_split=eval_split,
            generation_update_lock=self.generation_update_lock,
            rollout_queue=self.rollout_queue,
            rollout_slots=self.rollout_slots,
            **generate_kwargs,
        )
        # train
        self.trainer_actor = TrainingActor.remote(
            pretrain=pretrain,
            strategy=strategy,
            actor_model_group=actor_model_group,
            critic_model_group=critic_model_group,
            reward_model_group=reward_model_group,
            reference_model_group=reference_model_group,
            vllm_engines=vllm_engines,
            generation_update_lock=self.generation_update_lock,
            rollout_queue=self.rollout_queue,
            rollout_slots=self.rollout_slots,
        )

    def fit(self) -> None:
        checkpoint_states = ray.get(self.trainer_actor.init_checkpoint_states.remote())
        # Restore step and epoch
        start_episode = checkpoint_states["episode"]
        global_step = checkpoint_states["global_step"]
        total_consumed_prompts = checkpoint_states["total_consumed_prompts"]
        # Keep vLLM weights and dataloader states in sync when resuming.
        if global_step > 0:
            ray.get(
                [
                    self.generator_actor.load_state_dict.remote(checkpoint_states["data_loader_state_dict"]),
                    self.trainer_actor.broadcast_to_vllm.remote(),
                ]
            )

        # Launch async training
        ray.get(
            [
                self.generator_actor.fit.remote(episode=start_episode, total_consumed_prompts=total_consumed_prompts),
                self.trainer_actor.fit.remote(global_step=global_step),
            ]
        )

    def get_max_steps(self):
        return ray.get(self.generator_actor.get_max_steps.remote())
