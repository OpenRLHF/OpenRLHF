import os
import time
from abc import ABC
from datetime import timedelta
from typing import Dict, Tuple

import ray
import torch
from tqdm import tqdm

from openrlhf.datasets import PromptDataset
from openrlhf.datasets.utils import blending_datasets
from openrlhf.trainer.ppo_utils.experience_maker import RemoteExperienceMaker, SamplesGenerator
from openrlhf.trainer.ppo_utils.kl_controller import AdaptiveKLController, FixedKLController
from openrlhf.trainer.ppo_utils.replay_buffer import balance_experiences
from openrlhf.trainer.ray.launcher import RayActorGroup
from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.utils.logging_utils import TensorboardLogger, WandbLogger, init_logger
from openrlhf.utils.utils import get_tokenizer

logger = init_logger(__name__)


def prepare_datasets(strategy, tokenizer):
    args = strategy.args

    # prepare datasets
    train_data = blending_datasets(
        args.prompt_data,
        args.prompt_data_probs,
        strategy,
        args.seed,
        max_count=args.max_samples,
        dataset_split=args.prompt_split,
    )

    # Create train dataset
    train_data = train_data.select(range(min(args.max_samples, len(train_data))))
    prompts_dataset = PromptDataset(train_data, tokenizer, strategy, input_template=args.input_template)
    prompts_dataloader = strategy.setup_dataloader(prompts_dataset, 1, True, True, prompts_dataset.collate_fn)

    # Create eval dataset if eval data exists
    if getattr(args, "eval_dataset", None):
        eval_data = blending_datasets(
            args.eval_dataset,
            None,  # No probability sampling for eval datasets
            strategy,
            dataset_split=args.eval_split,
        )
        eval_data = eval_data.select(range(min(args.max_samples, len(eval_data))))
        eval_dataset = PromptDataset(eval_data, tokenizer, strategy, input_template=args.input_template)
        eval_dataloader = strategy.setup_dataloader(eval_dataset, 1, True, False, eval_dataset.collate_fn)
    else:
        eval_dataloader = None

    max_steps = (
        len(prompts_dataset) * args.n_samples_per_prompt // args.train_batch_size * args.num_episodes * args.max_epochs
    )
    return prompts_dataloader, eval_dataloader, max_steps


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

        if self.args.kl_target:
            self.kl_ctl = AdaptiveKLController(self.args.init_kl_coef, self.args.kl_target, self.args.kl_horizon)
        else:
            self.kl_ctl = FixedKLController(self.args.init_kl_coef)

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

    def train_step(self, rollout_samples, global_step: int) -> Tuple[Dict, int]:
        # Turn raw rollouts into PPO-ready trajectories with rewards.
        experiences = self.experience_maker.make_experience_batch(rollout_samples)

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
        """Broadcast actor weights to vLLM engines.

        When vllm_enable_sleep is enabled, we use fine-grained control:
        1. Wake up only weights (not KV cache) to minimize GPU memory during weight sync
        2. Broadcast weights from actor model to vLLM
        3. Keep vLLM in weights-only state; KV cache will be woken up later before generation

        This approach reduces peak GPU memory during gradient sync by avoiding
        simultaneous allocation of both weights and KV cache.
        """
        if self.args.vllm_enable_sleep:
            # Wake up only weights for weight sync (not KV cache)
            # This avoids allocating KV cache memory during weight update
            batch_vllm_engine_call(self.vllm_engines, "wake_up", tags=["weights"])

        ray.get(self.actor_model_group.async_run_method(method_name="broadcast_to_vllm"))

        # NOTE: We keep vLLM in weights-only state after weight sync.
        # KV cache will be woken up before generation in SamplesGenerator.

    def save_logs_and_checkpoints(self, global_step: int, logs_dict=None, client_states=None) -> None:
        logs_dict = logs_dict or {}
        if global_step % self.args.logging_steps == 0:
            if self.wandb_logger:
                self.wandb_logger.log_train(global_step, logs_dict)
            if self.tensorboard_logger:
                self.tensorboard_logger.log_train(global_step, logs_dict)

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
        return {
            "episode": 0,
            "global_step": 0,
            "total_consumed_prompts": 0,
            "data_loader_state_dict": {},
        }


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
        **generate_kwargs,
    ) -> None:
        # get eval and save steps
        if strategy.args.eval_steps == -1:
            strategy.args.eval_steps = float("inf")  # do not evaluate
        if strategy.args.save_steps == -1:
            strategy.args.save_steps = float("inf")  # do not save ckpt

        # Tokenizer is shared across the sample generator and trainer to avoid duplicated loads.
        tokenizer = get_tokenizer(pretrain, None, "left", strategy, use_fast=not strategy.args.disable_fast_tokenizer)
        self.prompts_dataloader, self.eval_dataloader, self.max_steps = prepare_datasets(strategy, tokenizer)
        self.generate_kwargs = generate_kwargs

        # sample generation
        self.samples_generator = SamplesGenerator(
            strategy=strategy,
            prompts_dataloader=self.prompts_dataloader,
            eval_dataloader=self.eval_dataloader,
            tokenizer=tokenizer,
            vllm_engines=vllm_engines,
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

    def get_max_steps(self):
        return self.max_steps

    def fit(self) -> None:
        checkpoint_states = self.init_checkpoint_states()
        # Restore step and start_epoch
        start_episode = checkpoint_states["episode"]
        global_step = checkpoint_states["global_step"]
        total_consumed_prompts = checkpoint_states["total_consumed_prompts"]
        # Keep vLLM weights and dataloader states in sync when resuming.
        if global_step:
            self.broadcast_to_vllm()
            state_dict = checkpoint_states["data_loader_state_dict"]
            if state_dict:
                self.prompts_dataloader.load_state_dict(state_dict)

        for episode in range(start_episode, self.args.num_episodes):
            dataset_length = len(self.prompts_dataloader)
            pbar = tqdm(
                range(dataset_length),
                desc=f"Episode [{episode + 1}/{self.args.num_episodes}]",
                initial=total_consumed_prompts % max(dataset_length, 1),
            )
            while True:
                # Draw one mini-batch of prompts; stop when loader is exhausted.
                rollout_samples, filter_pass_rate, prompts_consumed, is_exhausted = (
                    self.samples_generator.generate_samples(**self.generate_kwargs)
                )
                total_consumed_prompts += prompts_consumed
                if is_exhausted:
                    break

                # Run PPO update on this batch and bump the global step counter.
                status, global_step = self.train_step(rollout_samples, global_step)

                # Add generated samples to status dictionary
                if self.args.dynamic_filtering:
                    status["dynamic_filtering_pass_rate"] = filter_pass_rate
                log_status = {k: v for k, v in status.items() if k not in ["generated_samples"]}
                logger.info(f"✨ Global step {global_step}: {log_status}")

                # logs/checkpoints
                client_states = {
                    "episode": episode,
                    "global_step": global_step,
                    "total_consumed_prompts": total_consumed_prompts,
                    "data_loader_state_dict": self.prompts_dataloader.state_dict(),
                }
                self.save_logs_and_checkpoints(global_step, status, client_states)

                # TODO: Add evaluation mechanism for PPO
                if global_step % self.args.eval_steps == 0 and self.eval_dataloader:
                    eval_generate_kwargs = self.generate_kwargs.copy()
                    eval_generate_kwargs["temperature"] = self.args.eval_temperature
                    eval_generate_kwargs["n_samples_per_prompt"] = self.args.eval_n_samples_per_prompt
                    self.evaluate(global_step, **eval_generate_kwargs)

                pbar.update(prompts_consumed)

        # Close trackers
        if self.wandb_logger:
            self.wandb_logger.close()
        if self.tensorboard_logger:
            self.tensorboard_logger.close()

    @torch.no_grad()
    def evaluate(self, global_step, **generate_kwargs):
        """Evaluate model performance on eval dataset."""
        start_time = time.time()
        logger.info(f"⏰ Evaluation start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # First collect all prompts and labels
        prompt_to_datasource = {}  # Dictionary to store mapping between prompts and their data sources
        for datasources, prompts, labels in self.eval_dataloader:
            # Create mapping for each prompt to its corresponding data source
            for prompt, datasource in zip(prompts, datasources):
                prompt_to_datasource[prompt] = datasource

        # Generate samples and calculate rewards
        samples_list = self.samples_generator.generate_eval_samples(**generate_kwargs)

        # duplicate prompts and labels for each sample
        all_prompts = sum([s.prompts for s in samples_list], [])

        n_samples_per_prompt = generate_kwargs["n_samples_per_prompt"]

        # Get rewards from samples, such as agent rewards or remote reward models
        rewards_list = []
        for samples in samples_list:
            rewards_list.append(samples.rewards)
        # Reshape rewards to (num_prompts, n_samples_per_prompt)
        rewards = torch.tensor(rewards_list).reshape(-1, n_samples_per_prompt)

        # Collect local statistics for each data source
        global_metrics = {}  # {datasource: {"pass{n_samples_per_prompt}": 0, "pass1": 0, "count": 0}}

        # Process rewards in chunks of n_samples_per_prompt
        num_prompts = len(all_prompts) // n_samples_per_prompt
        for i in range(num_prompts):
            # Get the original prompt (first one in the chunk)
            original_prompt = all_prompts[i * n_samples_per_prompt]
            datasource = prompt_to_datasource[original_prompt]  # Get corresponding data source using the mapping
            if datasource not in global_metrics:
                global_metrics[datasource] = {f"pass{n_samples_per_prompt}": 0, "pass1": 0, "count": 0}

            # Get rewards for this chunk
            chunk_rewards = rewards[i]

            # Calculate pass@k and pass@1
            if n_samples_per_prompt > 1:
                global_metrics[datasource][f"pass{n_samples_per_prompt}"] += chunk_rewards.max().float().item()
            global_metrics[datasource]["pass1"] += chunk_rewards.mean().float().item()
            global_metrics[datasource]["count"] += 1

        # Calculate global averages
        logs = {}
        for datasource, metrics in global_metrics.items():
            logs[f"eval_{datasource}_pass{n_samples_per_prompt}"] = (
                metrics[f"pass{n_samples_per_prompt}"] / metrics["count"]
            )
            logs[f"eval_{datasource}_pass1"] = metrics["pass1"] / metrics["count"]

        # Log to wandb/tensorboard
        if self.wandb_logger:
            self.wandb_logger.log_eval(global_step, logs)
        if self.tensorboard_logger:
            self.tensorboard_logger.log_eval(global_step, logs)

        end_time = time.time()
        duration = end_time - start_time
        time_str = str(timedelta(seconds=duration)).split(".")[0]
        logger.info(f"✨ Evaluation completed in {time_str}, global_step {global_step}, eval_metrics: {logs}")
