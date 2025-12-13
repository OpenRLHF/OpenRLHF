import asyncio
import time

import ray
from ray.util.queue import Queue
from tqdm import tqdm

from openrlhf.trainer.ppo_trainer import BasePPOTrainer
from openrlhf.trainer.ppo_utils.misc import normalize_interval_config
from openrlhf.trainer.ppo_utils.sample_maker import RemoteSampleGenerater
from openrlhf.trainer.ray.launcher import RayActorGroup
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.utils import get_tokenizer

logger = init_logger(__name__)


@ray.remote(num_cpus=0)
class AsyncController:
    """Bounded queue with simple backpressure logging between generator/trainer."""

    def __init__(
        self,
        queue_size: int = 1,
        log_interval: int = 10,
    ):
        self.queue = Queue(maxsize=queue_size)
        self.log_interval = log_interval

    def _wait_for_consumer(self):
        log_counter = 0
        while self.queue.full():
            if log_counter % self.log_interval == 0:
                logger.info(f"[async-controller] queue is full, waiting for consumer...")
            log_counter = (log_counter + 1) % self.log_interval
            time.sleep(1)

    def put(self, item):
        """Enqueue an item with simple backpressure."""
        self._wait_for_consumer()
        self.queue.put(item)
        return {"status": "enqueued", "queued": self.queue.qsize()}

    def get(self):
        return self.queue.get()

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

    def set_generating(self, allow_generating):
        """Set generation state.

        Args:
            is_generating: True to allow generation, False to block it
        """
        self._toggle_event(self.generating_event, allow_generating)

    def set_update_weights(self, allow_updating):
        """Set weight update state.

        Args:
            is_updating: True to allow weight updates, False to block it
        """
        self._toggle_event(self.update_weights_event, allow_updating)


@ray.remote
class GenerateSamplesActor:

    def __init__(
        self,
        pretrain,
        strategy,
        vllm_engines=None,
        prompt_split="train",
        eval_split="test",
        *,
        signal_actor=None,
        controller_actor=None,
        **generate_kwargs,
    ):
        tokenizer = get_tokenizer(pretrain, None, "left", strategy, use_fast=not strategy.args.disable_fast_tokenizer)

        self.strategy = strategy
        self.args = strategy.args

        self.train_sample_generater = RemoteSampleGenerater(
            strategy=strategy,
            tokenizer=tokenizer,
            vllm_engines=vllm_engines,
            dataset_split=prompt_split,
            generate_kwargs=generate_kwargs,
        )

        self.signal_actor = signal_actor
        self.controller_actor = controller_actor

    def get_max_steps(self):
        return self.train_sample_generater.max_steps

    def load_state_dict(self, state_dict):
        self.train_sample_generater.load_state_dict(state_dict)

    def fit(self, episode, global_step):
        for episode in range(episode, self.args.num_episodes):
            pbar = tqdm(
                range(len(self.train_sample_generater.prompts_dataloader)),
                desc=f"Episode [{episode + 1}/{self.args.num_episodes}]",
                # initial=global_step, # TODO
            )
            while True:
                # Pause generation while weights are being broadcasted to keep outputs consistent.
                ray.get(self.signal_actor.wait_generating.remote())
                ray.get(self.signal_actor.set_update_weights.remote(False))

                samples, pass_rate, prompts_used, is_exhausted = self.train_sample_generater.make_sample_batch()

                # Re-open weight updates now that the batch is ready (or we've hit the end).
                ray.get(self.signal_actor.set_update_weights.remote(True))
                if is_exhausted:
                    break

                # Send the produced batch to the controller for training.
                ray.get(
                    self.controller_actor.put.remote(
                        (samples, episode, self.train_sample_generater.state_dict(), pass_rate)
                    )
                )
                pbar.update(prompts_used)

        ray.get(self.controller_actor.put.remote("done"))


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
        vllm_engines=None,
        signal_actor=None,
        controller_actor=None,
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
        self.controller_actor = controller_actor

    def fit(self, global_step):
        while True:
            # Draw one mini-batch of prompts; stop when loader is exhausted.
            output = ray.get(self.controller_actor.get.remote())
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
        vllm_engines=None,
        prompt_split: str = "train",
        eval_split: str = "test",
        **generate_kwargs,
    ) -> None:
        args = strategy.args
        # get eval and save steps
        normalize_interval_config(args)

        queue_size = getattr(args, "async_queue_size", 1)
        if queue_size <= 0:
            raise ValueError(f"async_queue_size must be positive, got {queue_size}")
        logger.info(f"queue_size={queue_size}")
        self.controller_actor = AsyncController.remote(queue_size=queue_size)
        self.signal_actor = SignalActor.remote()

        # rollout
        self.generator_actor = GenerateSamplesActor.remote(
            pretrain=pretrain,
            strategy=strategy,
            vllm_engines=vllm_engines,
            prompt_split=prompt_split,
            eval_split=eval_split,
            signal_actor=self.signal_actor,
            controller_actor=self.controller_actor,
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
            signal_actor=self.signal_actor,
            controller_actor=self.controller_actor,
        )

    def fit(self) -> None:
        checkpoint_states = ray.get(self.trainer_actor.init_checkpoint_states.remote())[0]
        # Restore step and epoch
        global_step = checkpoint_states["global_step"]
        start_episode = checkpoint_states["episode"]
        # Keep vLLM weights and dataloader states in sync when resuming.
        if global_step:
            ray.get(
                [
                    self.trainer_actor.broadcast_to_vllm.remote(),
                    self.generator_actor.load_state_dict.remote(checkpoint_states["data_loader_state_dict"]),
                ]
            )

        # Launch async training
        ray.get(
            [
                self.generator_actor.fit.remote(episode=start_episode, global_step=global_step),
                self.trainer_actor.fit.remote(global_step=global_step),
            ]
        )

    def get_max_steps(self):
        return ray.get(self.generator_actor.get_max_steps.remote())
