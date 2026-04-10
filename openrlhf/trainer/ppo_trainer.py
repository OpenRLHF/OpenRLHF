import os
import time
from abc import ABC
from datetime import timedelta
from typing import Dict, Tuple

import ray
import torch
import transformers
from tqdm import tqdm

_TRANSFORMERS_V5 = int(transformers.__version__.split(".")[0]) >= 5

from openrlhf.datasets import PromptDataset
from openrlhf.datasets.utils import blending_datasets
from openrlhf.trainer.ppo_utils.experience import balance_experiences
from openrlhf.trainer.ppo_utils.experience_maker import RemoteExperienceMaker
from openrlhf.trainer.ppo_utils.kl_controller import AdaptiveKLController, FixedKLController
from openrlhf.trainer.ppo_utils.samples_generator import SamplesGenerator
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
    prompts_dataloader = strategy.setup_dataloader(
        prompts_dataset,
        1,
        True,
        True,
        prompts_dataset.collate_fn,
        num_workers=args.dataloader_num_workers,
    )

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
        eval_dataloader = strategy.setup_dataloader(
            eval_dataset,
            1,
            True,
            False,
            eval_dataset.collate_fn,
            num_workers=args.dataloader_num_workers,
        )
    else:
        eval_dataloader = None

    max_steps = (
        len(prompts_dataset) * args.n_samples_per_prompt // args.train_batch_size * args.num_episodes * args.max_epochs
    )
    return prompts_dataloader, eval_dataloader, max_steps


def compute_eval_metrics(eval_dataloader, samples_list, n_samples_per_prompt):
    """Compute pass@k eval metrics from generated samples.

    Shared by both sync and async evaluation paths.
    """
    if not samples_list:
        return {}

    prompt_to_datasource = {}
    for datasources, prompts, labels, _images in eval_dataloader:
        for prompt, datasource in zip(prompts, datasources):
            prompt_to_datasource[prompt] = datasource

    # Single pass: collect prompts, rewards, response_length, truncated
    all_prompts = []
    all_rewards = []
    all_response_lengths = []
    all_truncated = []
    for s in samples_list:
        all_prompts.extend(s.prompts)
        all_rewards.append(s.rewards)
        all_response_lengths.append(s.response_length.item() if s.response_length is not None else None)
        all_truncated.append(s.truncated.item() if s.truncated is not None else None)

    rewards = torch.tensor(all_rewards).reshape(-1, n_samples_per_prompt)

    metrics = {}
    for i in range(len(all_prompts) // n_samples_per_prompt):
        ds = prompt_to_datasource.get(all_prompts[i * n_samples_per_prompt], "unknown")
        if ds not in metrics:
            metrics[ds] = {f"pass{n_samples_per_prompt}": 0, "pass1": 0, "count": 0, "lengths": [], "truncated": []}
        chunk = rewards[i]
        if n_samples_per_prompt > 1:
            metrics[ds][f"pass{n_samples_per_prompt}"] += chunk.max().float().item()
        metrics[ds]["pass1"] += chunk.mean().float().item()
        metrics[ds]["count"] += 1

        start = i * n_samples_per_prompt
        for j in range(start, start + n_samples_per_prompt):
            if all_response_lengths[j] is not None:
                metrics[ds]["lengths"].append(all_response_lengths[j])
            if all_truncated[j] is not None:
                metrics[ds]["truncated"].append(all_truncated[j])

    logs = {}
    total_lengths = []
    total_truncated = []
    for ds, m in metrics.items():
        logs[f"eval_{ds}_pass{n_samples_per_prompt}"] = m[f"pass{n_samples_per_prompt}"] / m["count"]
        logs[f"eval_{ds}_pass1"] = m["pass1"] / m["count"]
        if m["lengths"]:
            logs[f"eval_{ds}_response_length_mean"] = sum(m["lengths"]) / len(m["lengths"])
            total_lengths.extend(m["lengths"])
        if m["truncated"]:
            logs[f"eval_{ds}_truncated_rate"] = sum(m["truncated"]) / len(m["truncated"])
            total_truncated.extend(m["truncated"])

    if total_lengths:
        logs["eval_response_length_mean"] = sum(total_lengths) / len(total_lengths)
    if total_truncated:
        logs["eval_truncated_rate"] = sum(total_truncated) / len(total_truncated)
    logs["eval_num_samples"] = float(len(all_prompts))

    return logs


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

        # Best eval metric tracking
        self.best_eval_metric_value = float("-inf")
        self.best_eval_metric_key = getattr(self.args, "best_metric_key", "") or ""
        self._latest_eval_metric_value = None

    def restore_best_checkpoint_state(self, checkpoint_states) -> None:
        if not checkpoint_states:
            return

        checkpoint_metric_key = checkpoint_states.get("best_eval_metric_key")
        checkpoint_metric_value = checkpoint_states.get("best_eval_metric_value")

        if checkpoint_metric_key:
            self.best_eval_metric_key = checkpoint_metric_key
        if checkpoint_metric_value is not None:
            self.best_eval_metric_value = checkpoint_metric_value
            self._latest_eval_metric_value = checkpoint_metric_value

    def fit(self, global_step: int = 0) -> None:
        raise NotImplementedError("fit method is not implemented")

    def train_step(self, rollout_samples, global_step: int) -> Tuple[Dict, int]:
        # Turn raw rollouts into PPO-ready trajectories with rewards.
        t0 = time.time()
        experiences = self.experience_maker.make_experience_batch(rollout_samples)
        make_experience_time = time.time() - t0

        # Peek at the first decoded sample for quick sanity check.
        _decode = self.tokenizer.decode if _TRANSFORMERS_V5 else self.tokenizer.batch_decode
        sample0 = [
            _decode(experiences[0].sequences[0].unsqueeze(0), skip_special_tokens=True)[0],
            experiences[0].info["reward"][0].item(),
        ]
        logger.info(f"Sample: {sample0}")

        # Compute ground-truth rollout stats BEFORE dynamic batch splitting
        rollout_stats = self._compute_rollout_stats(experiences)

        # Balance experiences across DP ranks if needed.
        if self.args.use_dynamic_batch:
            experiences = balance_experiences(experiences, self.args)

        # Push experiences to actor (and critic) shards before PPO.
        refs = self.actor_model_group.async_run_method_batch(method_name="append", experience=experiences)
        if self.critic_model_group is not None:
            refs.extend(self.critic_model_group.async_run_method_batch(method_name="append", experience=experiences))
        ray.get(refs)

        # Perform PPO optimization for actor/critic and gather metrics.
        t0 = time.time()
        status = self.ppo_train(global_step)
        ppo_train_time = time.time() - t0

        # Sync weights to vLLM.
        t0 = time.time()
        if self.vllm_engines is not None:
            self.broadcast_to_vllm()
        broadcast_time = time.time() - t0

        # Refresh KL controller with the latest measurement.
        if "kl" in status:
            # TODO: KL controller must be FixedKLController; AdaptiveKLController is incompatible here.
            self.kl_ctl.update(status["kl"], self.args.rollout_batch_size * self.args.n_samples_per_prompt)

        # Per-phase timing breakdown
        status["timing/make_experience"] = make_experience_time
        status["timing/ppo_train"] = ppo_train_time
        status["timing/broadcast"] = broadcast_time

        # Merge rollout stats (ground-truth, pre-dynamic-batch)
        status.update(rollout_stats)

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

    def _detect_eval_metric_key(self, eval_metrics):
        """Auto-detect the eval metric key to track for best checkpoint.

        Returns None if best_metric_key is 'none' (disabled) or no suitable metric found.
        """
        if self.best_eval_metric_key == "none":
            return None  # Explicitly disabled
        if self.best_eval_metric_key:
            return self.best_eval_metric_key if self.best_eval_metric_key in eval_metrics else None
        # Auto-detect: prefer eval_*_pass1 metric
        for key in sorted(eval_metrics):
            if key.endswith("_pass1"):
                self.best_eval_metric_key = key
                return key
        return None

    def save_best_checkpoint(self, eval_metrics, global_step, client_states=None):
        """Save checkpoint if eval metric is the best so far.

        When best_metric_key is 'none' or auto-detection fails, this is a no-op
        (regular save_steps checkpoints still save the most recent).
        """
        if not eval_metrics:
            return

        metric_key = self._detect_eval_metric_key(eval_metrics)
        if metric_key is None or metric_key not in eval_metrics:
            return

        current_value = eval_metrics[metric_key]
        self._latest_eval_metric_value = current_value
        prev_best = self.best_eval_metric_value

        if current_value > self.best_eval_metric_value:
            self.best_eval_metric_value = current_value
            logger.info(
                f"New best eval metric: {metric_key}={current_value:.4f} at step {global_step} "
                f"(previous best: {prev_best if prev_best > float('-inf') else 'N/A'})"
            )

            client_states = client_states or {}
            client_states["best_eval_metric_key"] = metric_key
            client_states["best_eval_metric_value"] = current_value
            client_states["checkpoint_metric_key"] = metric_key

            tag = f"best_global_step{global_step}"
            refs = self.actor_model_group.async_run_method(
                method_name="save_checkpoint",
                tag=tag,
                client_states=client_states,
                metric_value=current_value,
                metric_key=metric_key,
            )
            if self.critic_model_group is not None:
                refs.extend(
                    self.critic_model_group.async_run_method(
                        method_name="save_checkpoint", tag=tag, metric_value=current_value, metric_key=metric_key
                    )
                )
            ray.get(refs)
            logger.info(f"Saved best checkpoint: {tag} ({metric_key}={current_value:.4f})")

    def save_logs_and_checkpoints(self, global_step: int, logs_dict=None, client_states=None) -> None:
        logs_dict = logs_dict or {}
        if global_step % self.args.logging_steps == 0:
            if self.wandb_logger:
                self.wandb_logger.log_train(global_step, logs_dict)
            if self.tensorboard_logger:
                self.tensorboard_logger.log_train(global_step, logs_dict)

        # save ckpt
        client_states = client_states or {}
        if global_step % self.args.save_steps == 0:
            tag = f"global_step{global_step}"
            metric_value = self._latest_eval_metric_value
            metric_key = client_states.get("checkpoint_metric_key") or self.best_eval_metric_key or None
            refs = self.actor_model_group.async_run_method(
                method_name="save_checkpoint",
                tag=tag,
                client_states=client_states,
                metric_value=metric_value,
                metric_key=metric_key,
            )
            if self.critic_model_group is not None:
                refs.extend(
                    self.critic_model_group.async_run_method(
                        method_name="save_checkpoint", tag=tag, metric_value=metric_value, metric_key=metric_key
                    )
                )
            ray.get(refs)

    def _compute_rollout_stats(self, experiences) -> Dict:
        """Compute ground-truth rollout statistics before dynamic batch splitting."""
        all_rewards = torch.cat([exp.info["reward"] for exp in experiences if "reward" in exp.info])
        all_response_lengths = torch.cat(
            [exp.response_length for exp in experiences if exp.response_length is not None]
        )
        all_truncated = torch.cat([exp.truncated for exp in experiences if exp.truncated is not None])

        stats = {
            "rollout/reward_mean": all_rewards.float().mean().item(),
            "rollout/reward_std": all_rewards.float().std().item() if len(all_rewards) > 1 else 0.0,
            "rollout/response_length_mean": all_response_lengths.float().mean().item(),
            "rollout/truncated_rate": all_truncated.float().mean().item(),
            "rollout/num_samples": float(len(all_rewards)),
        }
        return stats

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

    def fit(self, global_step: int = 0) -> None:
        checkpoint_states = self.init_checkpoint_states()
        self.restore_best_checkpoint_state(checkpoint_states)
        # Restore step and start_epoch
        start_episode = checkpoint_states["episode"]
        # Use checkpoint's global_step if resuming, otherwise use the parameter
        is_resuming = checkpoint_states["global_step"] > 0
        if is_resuming:
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
                t_gen_start = time.time()
                rollout_samples, filter_pass_rate, prompts_consumed, is_exhausted = (
                    self.samples_generator.generate_samples(**self.generate_kwargs)
                )
                generation_time = time.time() - t_gen_start
                total_consumed_prompts += prompts_consumed
                if is_exhausted:
                    break

                # Run PPO update on this batch and bump the global step counter.
                status, global_step = self.train_step(rollout_samples, global_step)

                # Generation timing (rollout via vLLM)
                status["timing/generation"] = generation_time
                status["timing/step_total"] = sum(v for k, v in status.items() if k.startswith("timing/"))

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

                # Evaluation and best checkpoint saving
                if global_step % self.args.eval_steps == 0 and self.eval_dataloader:
                    eval_generate_kwargs = self.generate_kwargs.copy()
                    eval_generate_kwargs["temperature"] = self.args.eval_temperature
                    eval_generate_kwargs["n_samples_per_prompt"] = self.args.eval_n_samples_per_prompt
                    eval_logs = self.evaluate(global_step, **eval_generate_kwargs)
                    self.save_best_checkpoint(eval_logs, global_step, client_states)

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

        samples_list = self.samples_generator.generate_eval_samples(**generate_kwargs)
        logs = compute_eval_metrics(self.eval_dataloader, samples_list, generate_kwargs["n_samples_per_prompt"])

        if self.wandb_logger:
            self.wandb_logger.log_eval(global_step, logs)
        if self.tensorboard_logger:
            self.tensorboard_logger.log_eval(global_step, logs)

        end_time = time.time()
        duration = end_time - start_time
        time_str = str(timedelta(seconds=duration)).split(".")[0]
        logger.info(f"✨ Evaluation completed in {time_str}, global_step {global_step}, eval_metrics: {logs}")
        return logs
