import asyncio

import ray
from ray.util.queue import Queue
from tqdm import tqdm

from openrlhf.trainer.ppo_trainer import BasePPOTrainer, prepare_datasets
from openrlhf.trainer.ppo_utils.experience_maker import SamplesGenerator
from openrlhf.trainer.ray.launcher import RayActorGroup
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.utils import get_tokenizer

logger = init_logger(__name__)


@ray.remote(num_cpus=0)
class VLLMLock:
    """Cross-actor mutex for vLLM critical section."""

    def __init__(self):
        self._lock = asyncio.Lock()

    async def acquire(self):
        await self._lock.acquire()

    async def release(self):
        self._lock.release()


@ray.remote
class GenerateSamplesActor:
    def __init__(
        self,
        pretrain,
        strategy,
        vllm_engines,
        *,
        vllm_lock,
        rollout_queue,
        rollout_slots,
        **generate_kwargs,
    ):
        self.args = strategy.args

        tokenizer = get_tokenizer(pretrain, None, "left", strategy, use_fast=not strategy.args.disable_fast_tokenizer)
        self.prompts_dataloader, self.eval_dataloader, self.max_steps = prepare_datasets(strategy, tokenizer)
        self.generate_kwargs = generate_kwargs

        self.samples_generator = SamplesGenerator(
            strategy=strategy,
            prompts_dataloader=self.prompts_dataloader,
            eval_dataloader=self.eval_dataloader,
            tokenizer=tokenizer,
            vllm_engines=vllm_engines,
        )

        self.vllm_lock = vllm_lock
        self.rollout_queue = rollout_queue
        # Token bucket: size == rollout_queue capacity.
        # Generator MUST take a token BEFORE generating; trainer returns token AFTER consuming.
        self.rollout_slots = rollout_slots

    def get_max_steps(self):
        return self.max_steps

    def load_state_dict(self, state_dict):
        self.prompts_dataloader.load_state_dict(state_dict)

    def fit(self, episode: int, total_consumed_prompts: int):
        for episode in range(episode, self.args.num_episodes):
            dataset_length = len(self.prompts_dataloader)
            pbar = tqdm(
                range(dataset_length),
                desc=f"Episode [{episode + 1}/{self.args.num_episodes}]",
                initial=total_consumed_prompts % max(dataset_length, 1),
            )
            while True:
                # Backpressure: only generate if we have queue capacity (token available).
                self.rollout_slots.get(block=True)

                # vLLM critical section: generation must not overlap with broadcast_to_vllm().
                ray.get(self.vllm_lock.acquire.remote())
                try:
                    rollout_samples, filter_pass_rate, prompts_consumed, is_exhausted = (
                        self.samples_generator.generate_samples(**self.generate_kwargs)
                    )
                    total_consumed_prompts += prompts_consumed
                finally:
                    ray.get(self.vllm_lock.release.remote())

                produced = bool(rollout_samples)
                if produced:
                    client_states = {
                        "episode": episode,
                        "total_consumed_prompts": total_consumed_prompts,
                        "data_loader_state_dict": self.prompts_dataloader.state_dict(),
                    }
                    self.rollout_queue.put((rollout_samples, client_states, filter_pass_rate), block=True)
                    if prompts_consumed:
                        pbar.update(prompts_consumed)
                else:
                    # Nothing enqueued => trainer will never "consume" this slot,
                    # so we must return the token here (prevents token leak / deadlock).
                    self.rollout_slots.put(None, block=True)

                if is_exhausted:
                    break

            pbar.close()

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
        *,
        vllm_lock,
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

        self.vllm_lock = vllm_lock
        self.rollout_queue = rollout_queue
        self.rollout_slots = rollout_slots

    def fit(self, global_step: int):
        while True:
            payload = self.rollout_queue.get(block=True)
            if payload == "done":
                break

            rollout_samples, client_states, filter_pass_rate = payload

            # Batch consumed => free one token to allow generator to produce next batch.
            self.rollout_slots.put(None, block=True)

            status, global_step = self.train_step(rollout_samples, global_step)

            if self.args.dynamic_filtering:
                status["dynamic_filtering_pass_rate"] = filter_pass_rate

            log_status = {k: v for k, v in status.items() if k not in ["generated_samples"]}
            logger.info(f"✨ Global step {global_step}: {log_status}")

            client_states.update({"global_step": global_step})
            self.save_logs_and_checkpoints(global_step, status, client_states)

        if self.wandb_logger:
            self.wandb_logger.close()
        if self.tensorboard_logger:
            self.tensorboard_logger.close()

    def broadcast_to_vllm(self):
        # vLLM critical section: must not overlap with generation.
        ray.get(self.vllm_lock.acquire.remote())
        try:
            super().broadcast_to_vllm()
        finally:
            ray.get(self.vllm_lock.release.remote())


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
        **generate_kwargs,
    ) -> None:
        # get eval and save steps
        if strategy.args.eval_steps == -1:
            strategy.args.eval_steps = float("inf")  # do not evaluate
        if strategy.args.save_steps == -1:
            strategy.args.save_steps = float("inf")  # do not save ckpt

        queue_size = getattr(strategy.args, "async_queue_size", 1)
        if queue_size <= 0:
            raise ValueError(f"async_queue_size must be positive, got {queue_size}")
        logger.info(f"queue_size={queue_size}")

        self.rollout_queue = Queue(maxsize=queue_size)

        # Token pool (counting semaphore) for queue capacity.
        self.rollout_slots = Queue(maxsize=queue_size)
        for _ in range(queue_size):
            self.rollout_slots.put(None, block=True)

        # Cross-actor mutex for vLLM critical section.
        self.vllm_lock = VLLMLock.remote()

        self.generator_actor = GenerateSamplesActor.remote(
            pretrain=pretrain,
            strategy=strategy,
            vllm_engines=vllm_engines,
            vllm_lock=self.vllm_lock,
            rollout_queue=self.rollout_queue,
            rollout_slots=self.rollout_slots,
            **generate_kwargs,
        )

        self.trainer_actor = TrainingActor.remote(
            pretrain=pretrain,
            strategy=strategy,
            actor_model_group=actor_model_group,
            critic_model_group=critic_model_group,
            reward_model_group=reward_model_group,
            reference_model_group=reference_model_group,
            vllm_engines=vllm_engines,
            vllm_lock=self.vllm_lock,
            rollout_queue=self.rollout_queue,
            rollout_slots=self.rollout_slots,
        )

    def fit(self) -> None:
        checkpoint_states = ray.get(self.trainer_actor.init_checkpoint_states.remote())

        # Restore step and epoch
        start_episode = checkpoint_states["episode"]
        global_step = checkpoint_states["global_step"]
        total_consumed_prompts = checkpoint_states.get("total_consumed_prompts", 0)
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
