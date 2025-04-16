import itertools
import time
from abc import ABC
from datetime import timedelta
from typing import Any, Callable, Optional

import torch

from openrlhf.datasets import PromptDataset, SFTDataset
from openrlhf.trainer.ppo_utils import AdaptiveKLController, FixedKLController, RemoteExperienceMaker
from openrlhf.trainer.ray.launcher import PPORayActorGroup
from openrlhf.utils import blending_datasets
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.utils.distributed_sampler import DistributedSampler
from openrlhf.utils.remote_rm_utils import remote_rm_fn_ray


class PPOTrainer(ABC):
    """
    Trainer for Proximal Policy Optimization (PPO) algorithm.
    """

    def __init__(
        self,
        strategy: DeepspeedStrategy,
        actor_model_group: PPORayActorGroup,
        critic_model_group: PPORayActorGroup,
        reward_model_group: PPORayActorGroup,
        reference_model_group: PPORayActorGroup,
        init_kl_coef: float = 0.001,
        kl_target: float = None,
        kl_horizon: int = 10000,
        max_epochs: int = 1,
        tokenizer: Optional[Callable[[Any], dict]] = None,
        prompt_max_len: int = 128,
        dataloader_pin_memory: bool = True,
        remote_rm_url: str = None,
        save_hf_ckpt: bool = False,
        disable_ds_ckpt: bool = False,
        **generate_kwargs,
    ) -> None:
        super().__init__()

        self.strategy = strategy
        self.actor_model_group = actor_model_group
        self.critic_model_group = critic_model_group
        self.reward_model_group = reward_model_group
        self.reference_model_group = reference_model_group
        self.max_epochs = max_epochs
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len
        self.dataloader_pin_memory = dataloader_pin_memory
        self.remote_rm_url = remote_rm_url
        self.save_hf_ckpt = save_hf_ckpt
        self.disable_ds_ckpt = disable_ds_ckpt
        self.generate_kwargs = generate_kwargs

        self.args = strategy.args
        self.freezing_actor_steps = getattr(self.args, "freezing_actor_steps", -1)

        if self.kl_target:
            self.kl_ctl = AdaptiveKLController(init_kl_coef, kl_target, kl_horizon)
        else:
            self.kl_ctl = FixedKLController(init_kl_coef)

        self.experience_maker = RemoteExperienceMaker(
            self.actor_model_group,
            self.critic_model_group,
            self.reward_model_group,
            self.reference_model_group,
            self.tokenizer,
            self.prompt_max_len,
            self.kl_ctl,
            self.strategy,
            self.remote_rm_url,
            vllm_engines=self.vllm_engines,
            packing_samples=self.strategy.args.packing_samples,
        )

    def fit(
        self,
        consumed_samples=0,
    ) -> None:
        args = self.args

        # Load datasets
        self.prepare_datasets()
        num_rollouts_per_episodes = len(self.prompts_dataset) // args.rollout_batch_size

        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = num_rollouts_per_episodes  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        # Restore step and start_epoch
        steps = consumed_samples // args.rollout_batch_size + 1
        start_episode = consumed_samples // args.rollout_batch_size // num_rollouts_per_episodes
        consumed_samples = consumed_samples % (num_rollouts_per_episodes * args.rollout_batch_size)

        for episode in range(start_episode, args.num_episodes):
            if isinstance(self.prompts_dataloader.sampler, DistributedSampler):
                self.prompts_dataloader.sampler.set_epoch(
                    episode, consumed_samples=0 if episode > start_episode else consumed_samples
                )
            pbar = tqdm(
                range(self.prompts_dataloader.__len__()),
                desc=f"Episode [{episode + 1}/{args.num_episodes}]",
                disable=not self.strategy.is_rank_0(),
            )

            for _, rand_prompts, labels in self.prompts_dataloader:
                for i, experience in enumerate(
                    self.experience_maker.make_experience_list(rand_prompts, labels, **self.generate_kwargs)
                ):
                    if i == 0:
                        output = self.tokenizer.batch_decode(
                            experience.sequences[0].unsqueeze(0), skip_special_tokens=True
                        )
                        self.strategy.print(output)
                    self.replay_buffer.append(experience)

                if self.args.advantage_estimator not in ["group_norm", "dr_grpo"]:
                    self.replay_buffer.normalize(
                        self.strategy, "advantages", divide_by_std=not self.args.no_advantage_std_norm
                    )
                status = self.ppo_train(steps)
                self.replay_buffer.clear()

                if "kl" in status:
                    self.kl_ctl.update(status["kl"], args.rollout_batch_size * args.n_samples_per_prompt)
                pbar.set_postfix(status)

                # logs/checkpoints
                client_states = {"consumed_samples": steps * args.rollout_batch_size}
                self.save_logs_and_checkpoints(args, steps, pbar, status, client_states)

                pbar.update()
                steps = steps + 1

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()

    def prepare_datasets(self):
        strategy = self.strategy
        args = self.strategy.args

        # prepare datasets
        train_data = blending_datasets(
            args.prompt_data,
            args.prompt_data_probs,
            strategy,
            args.seed,
            max_count=args.max_samples,
        )

        # Create train dataset
        train_data = train_data.select(range(min(args.max_samples, len(train_data))))
        self.prompts_dataset = PromptDataset(train_data, self.tokenizer, strategy, input_template=args.input_template)
        self.prompts_dataloader = strategy.setup_dataloader(
            self.prompts_dataset,
            args.rollout_batch_size // (strategy.world_size // strategy.ring_attn_size),
            True,
            True,
        )

        # Create eval dataset if eval data exists
        if getattr(args, "eval_dataset", None):
            eval_data = blending_datasets(
                args.eval_dataset,
                None,  # No probability sampling for eval datasets
                strategy,
            )
            eval_data = eval_data.select(range(min(args.max_samples, len(eval_data))))
            eval_dataset = PromptDataset(eval_data, self.tokenizer, strategy, input_template=args.input_template)
            self.eval_dataloader = strategy.setup_dataloader(eval_dataset, 1, True, False)
        else:
            self.eval_dataloader = None

        if args.pretrain_data:
            pretrain_data = blending_datasets(
                args.pretrain_data,
                args.pretrain_data_probs,
                strategy,
                args.seed,
            )
            pretrain_max_len = args.max_len if args.max_len else args.prompt_max_len + args.generate_max_len
            pretrain_dataset = SFTDataset(
                pretrain_data.select(
                    range(
                        min(
                            len(pretrain_data), args.max_epochs * len(self.prompts_dataset) * args.n_samples_per_prompt
                        )
                    )
                ),
                self.tokenizer,
                pretrain_max_len,
                strategy,
                pretrain_mode=True,
            )
            self.pretrain_dataloader = itertools.cycle(
                iter(
                    strategy.setup_dataloader(
                        pretrain_dataset,
                        args.micro_train_batch_size,
                        True,
                        True,
                        collate_fn=pretrain_dataset.collate_fn,
                    )
                )
            )
        else:
            self.pretrain_dataloader = None

    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, client_states={}):
        if global_step % args.logging_steps == 0:
            # wandb
            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {
                    "train/%s" % k: v
                    for k, v in {
                        **logs_dict,
                        "global_step": global_step,
                    }.items()
                }
                if self.experience_maker.perf_stats is not None:
                    logs.update({f"perf/experience_maker/{k}": v for k, v in self.experience_maker.perf_stats.items()})
                self._wandb.log(logs)
            # TensorBoard
            elif self._tensorboard is not None and self.strategy.is_rank_0():
                for k, v in logs_dict.items():
                    self._tensorboard.add_scalar(f"train/{k}", v, global_step)
                if self.experience_maker.perf_stats is not None:
                    for k, v in self.experience_maker.perf_stats.items():
                        self._tensorboard.add_scalar(f"perf/experience_maker/{k}", v, global_step)

        # TODO: Add evaluation mechanism for PPO
        if global_step % args.eval_steps == 0 and self.eval_dataloader and len(self.eval_dataloader) > 0:
            self.evaluate(self.eval_dataloader, global_step, args.eval_temperature, args.eval_n_samples_per_prompt)
        # save ckpt
        # TODO: save best model on dev, use loss/perplexity/others on whole dev dataset as metric
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            self._save_checkpoint(args, tag, client_states)

    def evaluate(self, eval_dataloader, global_step, temperature=0.6, n_samples_per_prompt=1):
        """Evaluate model performance on eval dataset.

        Args:
            eval_dataloader: DataLoader containing evaluation prompts, labels and data sources
            global_step: Current training step for logging
            n_samples_per_prompt: Number of samples to generate per prompt for pass@k calculation
        """
        start_time = time.time()
        if self.strategy.is_rank_0():
            logger.info(f"⏰ Evaluation start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # vLLM wakeup when vllm_enable_sleep
        if self.strategy.args.vllm_enable_sleep:
            from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call

            batch_vllm_engine_call(self.vllm_engines, "wake_up")
            torch_dist_barrier_and_cuda_sync()

        # Only run evaluation on ring attention rank0
        if self.strategy.ring_attn_group is None or self.strategy.ring_attn_rank == 0:

            with torch.no_grad():
                # First collect all prompts and labels
                all_prompts = []
                all_labels = []
                all_datasources = []

                for datasources, prompts, labels in eval_dataloader:
                    all_prompts.extend(prompts)
                    all_labels.extend(labels)
                    all_datasources.extend(datasources)

                # Generate samples and calculate rewards
                generate_kwargs = self.generate_kwargs.copy()
                generate_kwargs["temperature"] = temperature
                generate_kwargs["n_samples_per_prompt"] = n_samples_per_prompt
                samples = self.experience_maker.generate_samples(all_prompts, all_labels, **generate_kwargs)
                queries = [self.tokenizer.batch_decode(seq, skip_special_tokens=False) for seq in samples.sequences]

                # duplicate prompts and labels for each sample
                all_prompts = sum([[prompt] * n_samples_per_prompt for prompt in all_prompts], [])
                all_labels = sum([[label] * n_samples_per_prompt for label in all_labels], [])

                # Calculate rewards
                if self.experience_maker.custom_reward_func:
                    rewards = self.experience_maker.custom_reward_func.remote(queries, all_prompts, all_labels)
                else:
                    rank = torch.distributed.get_rank() // self.strategy.ring_attn_size
                    rm = self.remote_rm_url[rank % len(self.remote_rm_url)]
                    rewards = remote_rm_fn_ray.remote(rm, queries=queries, prompts=all_prompts, labels=all_labels)
                rewards = ray.get(rewards)

                # Reshape rewards to (num_prompts, n_samples_per_prompt)
                rewards = rewards.reshape(-1, n_samples_per_prompt)

                # Collect local statistics for each data source
                local_metrics = {}  # {datasource: {"pass{n_samples_per_prompt}": 0, "pass1": 0, "count": 0}}

                for i, datasource in enumerate(all_datasources):
                    if datasource not in local_metrics:
                        local_metrics[datasource] = {f"pass{n_samples_per_prompt}": 0, "pass1": 0, "count": 0}

                    # Calculate pass@k and pass@1
                    prompt_rewards = rewards[i]
                    local_metrics[datasource][f"pass{n_samples_per_prompt}"] += prompt_rewards.max().float().item()
                    local_metrics[datasource]["pass1"] += prompt_rewards.mean().float().item()
                    local_metrics[datasource]["count"] += 1

                # All gather metrics from all ranks
                gathered_metrics = [None] * (self.strategy.world_size // self.strategy.ring_attn_size)
                if self.strategy.ring_attn_group is not None:
                    # Only rank 0 in ring attention group gathers metrics
                    torch.distributed.all_gather_object(
                        gathered_metrics, local_metrics, group=self.experience_maker.ring_rank0_group
                    )
                else:
                    torch.distributed.all_gather_object(gathered_metrics, local_metrics)

                # Only rank0 processes the gathered metrics
                if self.strategy.is_rank_0():
                    # Combine metrics from all ranks
                    global_metrics = {}
                    for rank_metrics in gathered_metrics:
                        for datasource, metrics in rank_metrics.items():
                            if datasource not in global_metrics:
                                global_metrics[datasource] = {f"pass{n_samples_per_prompt}": 0, "pass1": 0, "count": 0}
                            global_metrics[datasource][f"pass{n_samples_per_prompt}"] += metrics[
                                f"pass{n_samples_per_prompt}"
                            ]
                            global_metrics[datasource]["pass1"] += metrics["pass1"]
                            global_metrics[datasource]["count"] += metrics["count"]

                    # Calculate global averages
                    logs = {}
                    for datasource, metrics in global_metrics.items():
                        logs[f"eval_{datasource}_pass{n_samples_per_prompt}"] = (
                            metrics[f"pass{n_samples_per_prompt}"] / metrics["count"]
                        )
                        logs[f"eval_{datasource}_pass1"] = metrics["pass1"] / metrics["count"]

                    # Log to wandb/tensorboard
                    if self._wandb is not None:
                        logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": global_step}.items()}
                        self._wandb.log(logs)
                    elif self._tensorboard is not None:
                        for k, v in logs.items():
                            self._tensorboard.add_scalar(f"eval/{k}", v, global_step)

        if self.strategy.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "sleep")

        torch.cuda.empty_cache()

        end_time = time.time()
        duration = end_time - start_time
        if self.strategy.is_rank_0():
            time_str = str(timedelta(seconds=duration)).split(".")[0]
            logger.info(f"✨ Evaluation completed in {time_str}")
