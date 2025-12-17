import asyncio

import ray
from tqdm import tqdm

from openrlhf.trainer.ppo_trainer import BasePPOTrainer
from openrlhf.trainer.ppo_utils.sample_maker import RemoteSampleGenerator
from openrlhf.trainer.ray.launcher import RayActorGroup
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.utils import get_tokenizer

logger = init_logger(__name__)


@ray.remote(num_cpus=0)
class BufferActor:
    """Bounded queue with simple backpressure logging between generator/trainer."""

    def __init__(self, queue_size: int = 1, log_interval: int = 10):
        self.queue = asyncio.Queue(maxsize=queue_size)
        self.log_interval = log_interval

    async def wait_for_consumer(self):
        log_counter = 0
        while self.queue.full():
            if log_counter % self.log_interval == 0:
                logger.info("[async-controller] queue is full, waiting for consumer...")
            log_counter = (log_counter + 1) % self.log_interval
            await asyncio.sleep(1)

    async def put(self, item):
        await self.wait_for_consumer()
        await self.queue.put(item)
        return {"status": "enqueued", "queued": self.queue.qsize()}

    async def get(self):
        return await self.queue.get()

    def stats(self):
        return {"queued": self.queue.qsize()}


@ray.remote(num_cpus=0)
class SignalActor:
    """Lightweight gatekeeper to coordinate generation and weight updates."""

    def __init__(self):
        self.generating_event = asyncio.Event()
        self.update_weights_event = asyncio.Event()
        self.set_generating(True)  # Initially allow generation
        self.set_update_weights(True)  # Initially allow weight updates

    @staticmethod
    def _toggle_event(event: asyncio.Event, allow: bool):
        event.set() if allow else event.clear()

    async def wait_generating(self):
        """Wait for generation to be allowed."""
        return await self.generating_event.wait()

    async def wait_update_weights(self):
        """Wait for weight update to be allowed."""
        return await self.update_weights_event.wait()

    def set_generating(self, allow_generating: bool):
        """Set generation state."""
        self._toggle_event(self.generating_event, allow_generating)

    def set_update_weights(self, allow_updating: bool):
        """Set weight update state."""
        self._toggle_event(self.update_weights_event, allow_updating)


@ray.remote
class RolloutActor:
    def __init__(
        self,
        pretrain,
        strategy,
        vllm_engines,
        prompt_split="train",
        eval_split="test",
        *,
        signal_actor,
        buffer_actor,
        **generate_kwargs,
    ):
        tokenizer = get_tokenizer(pretrain, None, "left", strategy, use_fast=not strategy.args.disable_fast_tokenizer)

        self.strategy = strategy
        self.args = strategy.args

        self.train_sample_generator = RemoteSampleGenerator(
            strategy=strategy,
            tokenizer=tokenizer,
            vllm_engines=vllm_engines,
            prompt_split=prompt_split,
            generate_kwargs=generate_kwargs,
        )

        self.signal_actor = signal_actor
        self.buffer_actor = buffer_actor

    def get_max_steps(self):
        return self.train_sample_generator.max_steps

    def load_state_dict(self, state_dict):
        self.train_sample_generator.load_state_dict(state_dict)

    def fit(self, episode, global_step):
        for episode in range(episode, self.args.num_episodes):
            pbar = tqdm(
                range(len(self.train_sample_generator.prompts_dataloader)),
                desc=f"Episode [{episode + 1}/{self.args.num_episodes}]",
                # initial=global_step, # TODO
            )
            while True:
                # Wait until queue is not full
                ray.get(self.buffer_actor.wait_for_consumer.remote())

                # Pause generation while weights are being broadcasted to keep outputs consistent.
                ray.get(self.signal_actor.wait_generating.remote())
                ray.get(self.signal_actor.set_update_weights.remote(False))

                samples, pass_rate, prompts_used, is_exhausted = self.train_sample_generator.make_sample_batch()

                # Re-open weight updates now that the batch is ready (or we've hit the end).
                ray.get(self.signal_actor.set_update_weights.remote(True))
                if is_exhausted:
                    break

                # Send the produced batch to the controller for training.
                ray.get(
                    self.buffer_actor.put.remote(
                        (samples, episode, self.train_sample_generator.state_dict(), pass_rate)
                    )
                )
                pbar.update(prompts_used)

        ray.get(self.buffer_actor.put.remote("done"))


@ray.remote
class TrainActor(BasePPOTrainer):
    def __init__(
        self,
        pretrain,
        strategy,
        actor_model_group,
        critic_model_group,
        reward_model_group,
        reference_model_group,
        vllm_engines,
        signal_actor,
        buffer_actor,
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
        self.signal_actor = signal_actor
        self.buffer_actor = buffer_actor

    def fit(self, global_step):
        while True:
            # Draw one mini-batch of prompts; stop when loader is exhausted.
            output = ray.get(self.buffer_actor.get.remote())
            if output == "done":
                break
            samples, episode, data_loader_state_dict, pass_rate = output

            # Run PPO update on this batch and bump the global step counter.
            status, global_step = self.train_step(samples, global_step)

            # Add generated samples to status dictionary
            if self.args.dynamic_filtering:
                status["dynamic_filtering_pass_rate"] = pass_rate
            log_status = {k: v for k, v in status.items() if k not in ["generated_samples"]}
            logger.info(f"✨ Global step {global_step}: {log_status}")

            # logs/checkpoints
            client_states = {
                "global_step": global_step,
                "episode": episode,
                "data_loader_state_dict": data_loader_state_dict,
            }
            self.save_logs_and_checkpoints(global_step, status, client_states)

        # Close trackers
        if self.wandb_logger:
            self.wandb_logger.close()
        if self.tensorboard_logger:
            self.tensorboard_logger.close()

    def broadcast_to_vllm(self):
        # pause generation for sync
        ray.get(self.signal_actor.set_generating.remote(False))
        ray.get(self.signal_actor.wait_update_weights.remote())
        # broadcast
        super().broadcast_to_vllm()
        # resume generation
        ray.get(self.signal_actor.set_generating.remote(True))


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
        if strategy.args.eval_steps == -1:
            strategy.args.eval_steps = float("inf")  # do not evaluate
        if strategy.args.save_steps == -1:
            strategy.args.save_steps = float("inf")  # do not save ckpt

        queue_size = getattr(args, "async_queue_size", 1)
        if queue_size <= 0:
            raise ValueError(f"async_queue_size must be positive, got {queue_size}")
        logger.info(f"queue_size={queue_size}")
        self.buffer_actor = BufferActor.remote(queue_size=queue_size)
        self.signal_actor = SignalActor.remote()

        # rollout
        self.rollout_actor = RolloutActor.remote(
            pretrain=pretrain,
            strategy=strategy,
            vllm_engines=vllm_engines,
            prompt_split=prompt_split,
            eval_split=eval_split,
            signal_actor=self.signal_actor,
            buffer_actor=self.buffer_actor,
            **generate_kwargs,
        )
        # train
        self.train_actor = TrainActor.remote(
            pretrain=pretrain,
            strategy=strategy,
            actor_model_group=actor_model_group,
            critic_model_group=critic_model_group,
            reward_model_group=reward_model_group,
            reference_model_group=reference_model_group,
            vllm_engines=vllm_engines,
            signal_actor=self.signal_actor,
            buffer_actor=self.buffer_actor,
        )

    def fit(self) -> None:
        checkpoint_states = ray.get(self.train_actor.init_checkpoint_states.remote())
        # Restore step and epoch
        global_step = checkpoint_states["global_step"]
        start_episode = checkpoint_states["episode"]
        # Keep vLLM weights and dataloader states in sync when resuming.
        if global_step > 0:
            ray.get(
                [
                    self.rollout_actor.load_state_dict.remote(checkpoint_states["data_loader_state_dict"]),
                    self.train_actor.broadcast_to_vllm.remote(),
                ]
            )

        # Launch async training
        ray.get(
            [
                self.rollout_actor.fit.remote(episode=start_episode, global_step=global_step),
                self.train_actor.fit.remote(global_step=global_step),
            ]
        )

    def get_max_steps(self):
        return ray.get(self.rollout_actor.get_max_steps.remote())
