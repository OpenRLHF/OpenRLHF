import os
import time
from abc import ABC
from datetime import timedelta
from typing import Any, Callable, Optional

import ray
import torch
import tqdm

from openrlhf.datasets import PromptDataset
from openrlhf.trainer.ppo_utils import AdaptiveKLController, FixedKLController, RemoteExperienceMaker
from openrlhf.trainer.ray.launcher import PPORayActorGroup
from openrlhf.utils import blending_datasets
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.remote_rm_utils import remote_rm_fn_ray

logger = init_logger(__name__)


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

        # wandb/tensorboard setting
        self._wandb = None
        self._tensorboard = None
        if self.strategy.args.use_wandb:
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=self.strategy.args.use_wandb)
            wandb.init(
                entity=self.strategy.args.wandb_org,
                project=self.strategy.args.wandb_project,
                group=self.strategy.args.wandb_group,
                name=self.strategy.args.wandb_run_name,
                config=self.strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/epoch")
            wandb.define_metric("eval/*", step_metric="eval/epoch", step_sync=True)

        # Initialize TensorBoard writer if wandb is not available
        if self.strategy.args.use_tensorboard and self._wandb is None:
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(self.strategy.args.use_tensorboard, exist_ok=True)
            log_dir = os.path.join(self.strategy.args.use_tensorboard, self.strategy.args.wandb_run_name)
            self._tensorboard = SummaryWriter(log_dir=log_dir)

    def fit(
        self,
        prompts_dataloader,
        eval_dataloader,
    ) -> None:
        args = self.args
        self.prompts_dataloader = prompts_dataloader
        self.eval_dataloader = eval_dataloader

        # Load datasets
        num_rollouts_per_episodes = len(self.prompts_dataloader)

        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = num_rollouts_per_episodes  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        # broadcast init checkpoint to vllm
        ckpt_path = os.path.join(args.ckpt_path, "_actor")
        if args.load_checkpoint and os.path.exists(ckpt_path) and not self.vllm_engines is None:
            # vLLM wakeup when vllm_enable_sleep
            if self.strategy.args.vllm_enable_sleep:
                from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call

                batch_vllm_engine_call(self.vllm_engines, "wake_up")

            ref = self.actor_model_group.async_run_method(method_name="broadcast_to_vllm")
            ray.get(ref)

            # vLLM offload when vllm_enable_sleep
            if self.strategy.args.vllm_enable_sleep:
                batch_vllm_engine_call(self.vllm_engines, "sleep")

        # Restore step and start_epoch
        consumed_samples = ray.get(self.actor_model_group.async_run_method(method_name="get_consumed_samples"))[0]
        steps = consumed_samples // args.rollout_batch_size + 1
        start_episode = consumed_samples // args.rollout_batch_size // num_rollouts_per_episodes
        consumed_samples = consumed_samples % (num_rollouts_per_episodes * args.rollout_batch_size)

        for episode in range(start_episode, args.num_episodes):
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
                    refs = self.actor_model_group.async_run_method(method_name="append", experience=experience)
                    if self.critic_model_group is not None:
                        refs.append(
                            self.critic_model_group.async_run_method(method_name="append", experience=experience)
                        )
                    ray.get(refs)

                status = self.ppo_train(steps)

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

    def ppo_train(self, global_steps):
        status = {}

        # triger remote critic model training
        if self.critic_model_group is not None:
            # sync for deepspeed_enable_sleep
            if self.strategy.args.deepspeed_enable_sleep:
                ray.get(self.critic_model_group.async_run_method(method_name="reload_states"))

            critic_status_ref = self.critic_model_group.async_run_method(method_name="fit")

            if self.strategy.args.colocate_all_models or self.strategy.args.deepspeed_enable_sleep:
                status.update(ray.get(critic_status_ref)[0])
            if self.strategy.args.deepspeed_enable_sleep:
                ray.get(self.critic_model_group.async_run_method(method_name="offload_states"))

        # actor model training
        if global_steps > self.freezing_actor_steps:
            if self.strategy.args.deepspeed_enable_sleep:
                self.actor_model_group.async_run_method(method_name="reload_states")

            actor_status_ref = self.actor_model_group.async_run_method(method_name="fit", global_steps=global_steps)
            status.update(ray.get(actor_status_ref)[0])

            if self.strategy.args.deepspeed_enable_sleep:
                self.actor_model_group.async_run_method(method_name="offload_states")

            # 4. broadcast weights to vllm engines
            if self.vllm_engines is not None:
                if self.strategy.args.vllm_enable_sleep:
                    from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call

                    batch_vllm_engine_call(self.vllm_engines, "wake_up")

                ray.get(self.actor_model_group.async_run_method(method_name="broadcast_to_vllm"))

                if self.strategy.args.vllm_enable_sleep:
                    batch_vllm_engine_call(self.vllm_engines, "sleep")

        # 5. wait remote critic model training done
        if self.critic_model_group and not self.strategy.args.colocate_all_models:
            status.update(ray.get(critic_status_ref)[0])

        return status

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
                self._wandb.log(logs)
            # TensorBoard
            elif self._tensorboard is not None and self.strategy.is_rank_0():
                for k, v in logs_dict.items():
                    self._tensorboard.add_scalar(f"train/{k}", v, global_step)

        # TODO: Add evaluation mechanism for PPO
        if global_step % args.eval_steps == 0 and self.eval_dataloader and len(self.eval_dataloader) > 0:
            self.evaluate(self.eval_dataloader, global_step, args.eval_temperature, args.eval_n_samples_per_prompt)
        # save ckpt
        # TODO: save best model on dev, use loss/perplexity/others on whole dev dataset as metric
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            ref = self.actor_model_group.async_run_method(
                method_name="save_checkpoint", args=args, tag=tag, client_states=client_states
            )
            if self.critic_model_group is not None:
                ref.append(
                    self.critic_model_group.async_run_method(
                        method_name="save_checkpoint", args=args, tag=tag, client_states=client_states
                    )
                )
            ray.get(ref)

    def evaluate(self, eval_dataloader, global_step, temperature=0.6, n_samples_per_prompt=1):
        """Evaluate model performance on eval dataset.

        Args:
            eval_dataloader: DataLoader containing evaluation prompts, labels and data sources
            global_step: Current training step for logging
            n_samples_per_prompt: Number of samples to generate per prompt for pass@k calculation
        """
        start_time = time.time()
        logger.info(f"⏰ Evaluation start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # vLLM wakeup when vllm_enable_sleep
        if self.strategy.args.vllm_enable_sleep:
            from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call

            batch_vllm_engine_call(self.vllm_engines, "wake_up")

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
            queries_list = [self.tokenizer.batch_decode(seq, skip_special_tokens=False) for seq in samples.sequences]

            # duplicate prompts and labels for each sample
            all_prompts = sum([[prompt] * n_samples_per_prompt for prompt in all_prompts], [])
            all_labels = sum([[label] * n_samples_per_prompt for label in all_labels], [])

            # Calculate rewards
            if self.experience_maker.custom_reward_func:
                # Let Ray automatically distribute the workload across available resources
                batch_size = self.strategy.args.micro_rollout_batch_size
                num_chunks = (len(queries_list) + batch_size - 1) // batch_size
                r_refs = []
                for i in range(num_chunks):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, len(queries_list))
                    r = self.custom_reward_func.remote(
                        queries_list[start_idx:end_idx],
                        all_prompts[start_idx:end_idx],
                        all_labels[start_idx:end_idx],
                    )
                    r_refs.append(r)
            else:
                # Distribute data across different remote reward function servers
                num_servers = len(self.remote_rm_url)
                batch_size = (len(queries_list) + num_servers - 1) // num_servers
                r_refs = []
                for i in range(num_servers):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, len(queries_list))
                    rm = self.remote_rm_url[i]
                    r = remote_rm_fn_ray.remote(
                        rm,
                        queries=queries_list[start_idx:end_idx],
                        prompts=all_prompts[start_idx:end_idx],
                        labels=all_labels[start_idx:end_idx],
                    )
                    r_refs.append(r)

            # Reshape rewards to (num_prompts, n_samples_per_prompt)
            rewards = rewards.reshape(-1, n_samples_per_prompt)

            # Collect local statistics for each data source
            global_metrics = {}  # {datasource: {"pass{n_samples_per_prompt}": 0, "pass1": 0, "count": 0}}

            for i, datasource in enumerate(all_datasources):
                if datasource not in global_metrics:
                    global_metrics[datasource] = {f"pass{n_samples_per_prompt}": 0, "pass1": 0, "count": 0}

                # Calculate pass@k and pass@1
                prompt_rewards = rewards[i]
                global_metrics[datasource][f"pass{n_samples_per_prompt}"] += prompt_rewards.max().float().item()
                global_metrics[datasource]["pass1"] += prompt_rewards.mean().float().item()
                global_metrics[datasource]["count"] += 1

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

        end_time = time.time()
        duration = end_time - start_time
        if self.strategy.is_rank_0():
            time_str = str(timedelta(seconds=duration)).split(".")[0]
            logger.info(f"✨ Evaluation completed in {time_str}")


def prepare_datasets(strategy, args, tokenizer):
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
    prompts_dataset = PromptDataset(train_data, tokenizer, strategy, input_template=args.input_template)
    prompts_dataloader = strategy.setup_dataloader(
        prompts_dataset,
        args.rollout_batch_size,
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
        eval_dataset = PromptDataset(eval_data, tokenizer, strategy, input_template=args.input_template)
        eval_dataloader = strategy.setup_dataloader(eval_dataset, 1, True, False)
    else:
        eval_dataloader = None

    return prompts_dataloader, eval_dataloader
