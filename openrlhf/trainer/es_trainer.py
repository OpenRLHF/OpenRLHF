import json
import random
import time
from collections import defaultdict
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import ray
import torch

from openrlhf.trainer.es_utils import checkpoints
from openrlhf.trainer.es_utils.data_adapter import (
    EVAL_SEED,
    STABILIZE_SEED,
    ESEvalSample,
    ESExperience,
    prepare_datasets,
    summarize_experience_metrics,
)
from openrlhf.trainer.es_utils.generator import ESSamplesGenerator
from openrlhf.trainer.ppo_trainer import compute_eval_metrics
from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call
from openrlhf.utils.logging_utils import TensorboardLogger, WandbLogger, init_logger
from openrlhf.utils.utils import get_tokenizer

logger = init_logger(__name__)
__all__ = [
    "ESTrainer",
    "ESExperience",
    "ESEvalSample",
    "ESSamplesGenerator",
    "STABILIZE_SEED",
    "EVAL_SEED",
    "prepare_datasets",
    "summarize_experience_metrics",
]


@ray.remote
class ESTrainer:
    """Evolutionary Strategies trainer backed by vLLM worker mutations."""

    def __init__(
        self,
        pretrain: str,
        strategy,
        actor_model_group,
        critic_model_group,
        reward_model_group,
        reference_model_group,
        vllm_engines: List,
        **generate_kwargs,
    ) -> None:
        if strategy.args.eval_steps == -1:
            strategy.args.eval_steps = float("inf")

        self.strategy = strategy
        self.args = strategy.args
        # ES can now consume either a legacy reward model group or a Ray reward graph manager.
        self.reward_model_group = reward_model_group
        self.vllm_engines = vllm_engines
        self.actor_model_group = actor_model_group
        self.critic_model_group = critic_model_group
        self.reference_model_group = reference_model_group

        tokenizer = get_tokenizer(pretrain, None, "left", strategy, use_fast=not strategy.args.disable_fast_tokenizer)
        self.tokenizer = tokenizer
        self.wandb_logger = WandbLogger(self.args) if self.args.use_wandb else None
        self.tensorboard_logger = TensorboardLogger(self.args) if self.args.use_tensorboard else None

        self.prompts_dataloader, self.eval_dataloader, self.steps_per_epoch = prepare_datasets(strategy, tokenizer)
        logger.info("Estimated ES steps per epoch (for logging): %s", self.steps_per_epoch)

        self.generate_kwargs = generate_kwargs
        self.es_std = self.args.es_std
        self.population_size = self.args.population_size
        self.es_shared_batch = self.args.es_shared_batch
        self._rng = random.Random(self.args.seed)

        self.samples_generator = ESSamplesGenerator(
            strategy=strategy,
            prompts_dataloader=self.prompts_dataloader,
            eval_dataloader=self.eval_dataloader,
            tokenizer=tokenizer,
            vllm_engines=vllm_engines,
            reward_model_group=self.reward_model_group,
        )

        self.pretrain = pretrain
        self._episode_idx = 0
        self.best_eval_metric_key = self.args.best_eval_metric_key
        self.best_eval_metric_value = float("-inf")
        self._latest_eval_metric_value = float("-inf")
        self._last_rollout_samples_for_tb: List[ESExperience] = []

    def _es_tb_text_sample_count(self) -> int:
        return max(0, int(getattr(self.args, "tensorboard_text_samples", 0)))

    def _es_tb_text_max_chars(self) -> int:
        return max(256, int(getattr(self.args, "tensorboard_text_max_chars", 12000)))

    @staticmethod
    def _es_tb_spaced_indices(n: int, k: int) -> List[int]:
        if k <= 0 or n <= 0:
            return []
        if n <= k:
            return list(range(n))
        denom = max(k - 1, 1)
        return [int(round(i * (n - 1) / denom)) for i in range(k)]

    @staticmethod
    def _es_tb_field_to_str(value: object) -> str:
        """Dataset prompt/label fields may be str, list (e.g. chat), or nested JSON."""
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (list, dict)):
            try:
                return json.dumps(value, ensure_ascii=False)
            except (TypeError, ValueError):
                return repr(value)
        return str(value)

    def _tensorboard_log_rollout_text(
        self,
        global_step: int,
        tag: str,
        samples: List[ESExperience],
        max_samples: int,
    ) -> None:
        if not self.tensorboard_logger or not samples or max_samples <= 0:
            return
        writer = self.tensorboard_logger.writer
        max_chars = self._es_tb_text_max_chars()
        idxs = self._es_tb_spaced_indices(len(samples), max_samples)
        blocks: List[str] = []
        for rank, idx in enumerate(idxs):
            sample = samples[idx]
            prompt = sample.prompts[0] if sample.prompts else ""
            label = sample.labels[0] if sample.labels else ""
            seed_val = int(sample.seeds[0].item()) if sample.seeds is not None and sample.seeds.numel() else None
            reward_s = (
                f"{sample.rewards[0].item():.6f}" if sample.rewards is not None and sample.rewards.numel() else "None"
            )
            extra_parts: List[str] = []
            if sample.info:
                for key, tensor in sorted(sample.info.items()):
                    if tensor is not None and tensor.numel():
                        extra_parts.append(f"{key}={tensor[0].item():.6f}")
            extras = ", ".join(extra_parts) if extra_parts else "(none)"
            seq = sample.sequences[0].tolist() if sample.sequences is not None else []
            try:
                full_decoded = self.tokenizer.decode(seq, skip_special_tokens=False) if seq else ""
            except Exception as exc:  # noqa: BLE001
                full_decoded = f"<decode_error {exc!r}>"
            prompt_s = self._es_tb_field_to_str(prompt)
            label_s = self._es_tb_field_to_str(label)
            prompt_c = prompt_s[:4000] + ("…" if len(prompt_s) > 4000 else "")
            label_c = label_s[:4000] + ("…" if len(label_s) > 4000 else "")
            full_c = full_decoded[:max_chars] + ("…" if len(full_decoded) > max_chars else "")
            blocks.append(
                f"#### Sample {rank + 1} (batch index {idx})  seed={seed_val}\n\n"
                f"**Reward:** {reward_s}  \n**Extra logs:** {extras}\n\n"
                f"**Label (raw):**  \n```\n{label_c}\n```\n\n"
                f"**Prompt (truncated):**  \n```\n{prompt_c}\n```\n\n"
                f"**Full decoded sequence (prompt + completion):**  \n```\n{full_c}\n```\n"
            )
        text = "\n\n---\n\n".join(blocks)
        text = text.replace("\x00", "")
        writer.add_text(tag, text, global_step)

    def _write_trainer_state(self, global_step: int, client_states: Dict) -> None:
        checkpoints.write_trainer_state(
            self.args.ckpt_path,
            global_step,
            self._episode_idx,
            self.best_eval_metric_key,
            self.best_eval_metric_value,
            client_states,
        )

    def _es_save_hf_checkpoint(self, tag: str, global_step: int, client_states: Dict) -> None:
        checkpoints.save_es_hf_checkpoint(
            args=self.args,
            tag=tag,
            global_step=global_step,
            client_states=client_states,
            vllm_engines=self.vllm_engines,
            pretrain=self.pretrain,
            episode_idx=self._episode_idx,
            best_eval_metric_key=self.best_eval_metric_key,
            best_eval_metric_value=self.best_eval_metric_value,
        )

    def _detect_eval_metric_key(self, eval_metrics: Dict) -> Optional[str]:
        metric_key, resolved_key = checkpoints.detect_eval_metric_key(self.best_eval_metric_key, eval_metrics)
        self.best_eval_metric_key = resolved_key
        return metric_key

    def save_best_checkpoint(self, eval_metrics: Dict, global_step: int, client_states=None) -> None:
        if not eval_metrics or not self.args.save_checkpoint:
            return
        if self.args.save_steps == -1:
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
                "New best eval metric: %s=%.4f at step %s (previous best: %s)",
                metric_key,
                current_value,
                global_step,
                prev_best if prev_best > float("-inf") else "N/A",
            )

            client_states = dict(client_states or {})
            client_states["best_eval_metric_key"] = metric_key
            client_states["best_eval_metric_value"] = current_value
            client_states["checkpoint_metric_key"] = metric_key

            self._es_save_hf_checkpoint(f"best_global_step{global_step}", global_step, client_states)
            logger.info("Saved best checkpoint: best_global_step%s (%s=%.4f)", global_step, metric_key, current_value)

    def init_checkpoint_states(self) -> Dict:
        state, best_key, best_value = checkpoints.init_checkpoint_states(
            self.args.load_checkpoint,
            self.args.ckpt_path,
        )
        if best_key is not None:
            self.best_eval_metric_key = best_key
        if best_value is not None:
            self.best_eval_metric_value = best_value
        return state

    def save_logs_and_checkpoints(self, global_step: int, logs_dict=None, client_states=None) -> None:
        logs_dict = logs_dict or {}
        client_states = client_states or {}
        if global_step % self.args.logging_steps == 0:
            if self.wandb_logger:
                self.wandb_logger.log_train(global_step, logs_dict)
            if self.tensorboard_logger:
                self.tensorboard_logger.log_train(global_step, logs_dict)
                n_tb = self._es_tb_text_sample_count()
                if n_tb and self._last_rollout_samples_for_tb:
                    self._tensorboard_log_rollout_text(
                        global_step,
                        "train/rollout_completions",
                        self._last_rollout_samples_for_tb,
                        n_tb,
                    )
                self.tensorboard_logger.writer.flush()

        if self.args.save_checkpoint and self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
            tag = f"global_step{global_step}"
            metric_value = self._latest_eval_metric_value
            metric_key = client_states.get("checkpoint_metric_key") or self.best_eval_metric_key
            checkpoint_state = dict(client_states)
            if metric_key is not None:
                checkpoint_state["checkpoint_metric_key"] = metric_key
            if metric_value is not None and metric_value > float("-inf"):
                checkpoint_state["metric_value"] = metric_value
            self._es_save_hf_checkpoint(tag, global_step, checkpoint_state)

        if self.actor_model_group is not None and self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
            refs = self.actor_model_group.async_run_method(
                method_name="save_checkpoint",
                tag=f"global_step{global_step}",
                client_states=client_states,
            )
            if self.critic_model_group is not None:
                refs.extend(
                    self.critic_model_group.async_run_method(
                        method_name="save_checkpoint", tag=f"global_step{global_step}"
                    )
                )
            ray.get(refs)

    def fit(self) -> None:
        checkpoint_states = self.init_checkpoint_states()
        global_step = checkpoint_states["global_step"]

        if global_step > 0 and self.actor_model_group is not None:
            if self.args.vllm_enable_sleep:
                batch_vllm_engine_call(self.vllm_engines, "wake_up", tags=["weights"])
            ray.get(self.actor_model_group.async_run_method(method_name="broadcast_to_vllm"))

        logger.info("Starting ES Training from step %s...", global_step)
        max_steps_cap = float("inf") if self.args.max_training_steps is None else float(self.args.max_training_steps)
        epoch = checkpoint_states.get("epoch", checkpoint_states.get("episode", 0))
        self._episode_idx = epoch

        if global_step == 0 and self.eval_dataloader:
            logger.info("Running step-0 baseline evaluation (before training)...")
            self.evaluate(global_step, **self.generate_kwargs)

        stop_training = False
        while epoch < self.args.max_epochs:
            self._episode_idx = epoch
            while True:
                if global_step >= max_steps_cap:
                    stop_training = True
                    break

                run_eval = bool(self.eval_dataloader and (global_step + 1) % self.args.eval_steps == 0)
                eval_start_time = time.time() if run_eval else None
                status, global_step, is_exhausted, eval_samples = self.train_step(global_step, include_eval=run_eval)
                if eval_samples is not None:
                    self._log_eval(global_step, eval_samples, eval_start_time, **self.generate_kwargs)
                if global_step % self.args.logging_steps == 0:
                    self.save_logs_and_checkpoints(global_step, status)
                    logger.info("Step %s: reward=%.4f", global_step, status.get("avg_reward", 0.0))
                if is_exhausted:
                    break

            if stop_training:
                break

            epoch += 1
            if epoch >= self.args.max_epochs:
                break
            self.samples_generator.reset_train_iterator()

        if self.args.save_checkpoint and self.args.save_steps == 0:
            logger.info(
                "save_steps=0; writing final checkpoint for global_step=%s to %s",
                global_step,
                self.args.ckpt_path,
            )
            self._es_save_hf_checkpoint(f"global_step{global_step}", global_step, {"final": True})

        if self.wandb_logger:
            self.wandb_logger.close()
        if self.tensorboard_logger:
            self.tensorboard_logger.close()

    def train_step(
        self, global_step: int, include_eval: bool = False
    ) -> Tuple[Dict, int, bool, Optional[List[ESExperience]]]:
        start_time = time.time()
        seeds = [self._rng.randint(0, 2**31 - 1) for _ in range(self.population_size)]
        if self.args.es_stabilize_seed:
            seeds[0] = STABILIZE_SEED

        rollout_start = time.time()
        all_samples, _, prompts_consumed, is_exhausted = self.samples_generator.generate_samples(
            engine_seeds=seeds,
            es_std=self.es_std,
            shared_batch=self.es_shared_batch,
            include_eval=include_eval,
            **self.generate_kwargs,
        )
        logger.info("Step %s: Population processing time: %.2fs", global_step, time.time() - rollout_start)

        eval_samples: Optional[List[ESExperience]] = None
        if include_eval:
            eval_samples = [s for s in all_samples if s.seeds is not None and EVAL_SEED in s.seeds.tolist()]
            rollout_samples = [s for s in all_samples if s.seeds is None or EVAL_SEED not in s.seeds.tolist()]
        else:
            rollout_samples = all_samples

        seed_scores: Dict[int, List[float]] = defaultdict(list)
        total_reward = 0.0
        for sample in rollout_samples:
            sample_seeds = sample.seeds.tolist() if sample.seeds is not None else []
            sample_rewards = [sample.rewards[0].item()] if sample.rewards is not None else []
            for seed_val, reward_val in zip(sample_seeds, sample_rewards):
                seed_scores[seed_val].append(reward_val)
                total_reward += reward_val

        updates = self._normalize_seed_scores(seed_scores)
        ray.get([engine.apply_es_gradient.remote(updates) for engine in self.vllm_engines])

        num_samples = sum(len(sample.seeds) if sample.seeds is not None else 0 for sample in rollout_samples)
        status = {
            "avg_reward": total_reward / max(1, num_samples),
            "num_samples": num_samples,
            "num_seeds": len(seed_scores),
            "step_time": time.time() - start_time,
            "prompts_consumed": prompts_consumed,
        }
        status.update(summarize_experience_metrics(rollout_samples, "train"))
        self._last_rollout_samples_for_tb = list(rollout_samples)
        return status, global_step + 1, is_exhausted, eval_samples

    def _normalize_seed_scores(self, seed_scores: Dict[int, List[float]]) -> List[Tuple[int, float, float]]:
        if not seed_scores:
            return []

        seed_means = {seed: np.mean(scores) for seed, scores in seed_scores.items()}
        scores_tensor = torch.tensor(list(seed_means.values()), dtype=torch.float32)
        mean = scores_tensor.mean()
        std = scores_tensor.std()
        if torch.isnan(std) or std < 1e-8:
            std = 1.0
        normalized = (scores_tensor - mean) / std
        return [
            (seed, norm_score, self.es_std) for (seed, _), norm_score in zip(seed_means.items(), normalized.tolist())
        ]

    def _log_eval(
        self, global_step: int, samples_list: List[ESExperience], start_time: float, **generate_kwargs
    ) -> None:
        n_tb = self._es_tb_text_sample_count()
        if self.tensorboard_logger and samples_list and n_tb > 0:
            self._tensorboard_log_rollout_text(
                global_step,
                "eval/validation_completions",
                samples_list,
                n_tb,
            )
            self.tensorboard_logger.writer.flush()

        eval_samples: List[ESEvalSample] = []
        for sample in samples_list:
            if sample.rewards is None or not sample.prompts:
                continue
            eval_samples.append(ESEvalSample(prompts=sample.prompts, rewards=sample.rewards[0].item()))

        if not eval_samples:
            logger.warning("No rewards collected during evaluation at step %s", global_step)
            return

        logs = compute_eval_metrics(
            self.eval_dataloader,
            eval_samples,
            generate_kwargs.get("n_samples_per_prompt", self.args.n_samples_per_prompt),
        )
        logs.update(summarize_experience_metrics(samples_list, "eval"))

        if self.wandb_logger:
            self.wandb_logger.log_eval(global_step, logs)
        if self.tensorboard_logger:
            self.tensorboard_logger.log_eval(global_step, logs)
            self.tensorboard_logger.writer.flush()
        if self.args.save_checkpoint and logs:
            self.save_best_checkpoint(logs, global_step)

        time_str = str(timedelta(seconds=time.time() - start_time)).split(".")[0]
        logger.info("Evaluation completed in %s, global_step %s, eval_metrics: %s", time_str, global_step, logs)

    @torch.no_grad()
    def evaluate(self, global_step, **generate_kwargs):
        start_time = time.time()
        logger.info("Evaluation start time: %s", time.strftime("%Y-%m-%d %H:%M:%S"))

        validation_start_time = time.time()
        samples_list = self.samples_generator.generate_eval_samples(**generate_kwargs)
        logger.info(
            "Step %s: Validation generation time: %.2fs",
            global_step,
            time.time() - validation_start_time,
        )

        self._log_eval(global_step, samples_list, start_time, **generate_kwargs)
