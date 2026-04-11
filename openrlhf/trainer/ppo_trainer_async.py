import math
import threading
import time
from collections import defaultdict, deque
from queue import Empty

import ray
from ray.util.queue import Queue
from tqdm import tqdm

from openrlhf.trainer.ppo_trainer import BasePPOTrainer, compute_eval_metrics, prepare_datasets
from openrlhf.trainer.ppo_utils.samples_generator import SamplesGenerator
from openrlhf.trainer.ray.launcher import RayActorGroup
from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.utils import get_tokenizer

logger = init_logger(__name__)


def _scalarize_metric(value, default: int | float = 0):
    if value is None:
        return default
    if hasattr(value, "numel"):
        if value.numel() == 0:
            return default
        return value.reshape(-1)[0].item()
    return value


def _summarize_rollout_versions(
    rollout_samples,
    *,
    current_param_version: int,
    fallback_param_version: int,
    stale_trajectory_processed: int,
):
    sample_min_versions = []
    sample_max_versions = []
    partial_old_token_count = 0

    for sample in rollout_samples:
        info = getattr(sample, "info", {}) or {}
        min_version = int(_scalarize_metric(info.get("min_weight_version"), fallback_param_version))
        max_version = int(_scalarize_metric(info.get("max_weight_version"), fallback_param_version))
        sample_min_versions.append(min_version)
        sample_max_versions.append(max_version)
        partial_old_token_count += int(_scalarize_metric(info.get("partial_old_token_count"), 0))

    if not sample_max_versions:
        version_gaps = [current_param_version - fallback_param_version]
        sample_min_versions = [fallback_param_version]
        sample_max_versions = [fallback_param_version]
    else:
        version_gaps = [current_param_version - version for version in sample_max_versions]

    stale_count = sum(gap >= 1 for gap in version_gaps)
    partial_spans = [max_v - min_v for min_v, max_v in zip(sample_min_versions, sample_max_versions, strict=False)]
    partial_count = sum(span > 0 for span in partial_spans)
    stale_trajectory_processed += stale_count

    metrics = {
        "fully_async/current_param_version": current_param_version,
        "fully_async/is_stale_batch": int(stale_count > 0),
        "fully_async/max_version_gap": max(version_gaps),
        "fully_async/min_sample_weight_version": min(sample_min_versions),
        "fully_async/max_sample_weight_version": max(sample_max_versions),
        "fully_async/count/stale_samples_processed": stale_count,
        "fully_async/count/stale_trajectory_processed": stale_trajectory_processed,
        "fully_async/partial/total_partial_num": partial_count,
        "fully_async/partial/partial_ratio": partial_count / len(sample_max_versions),
        "fully_async/partial/max_partial_span": max(partial_spans) if partial_spans else 0,
        "fully_async/partial/old_token_count": partial_old_token_count,
    }
    return metrics, stale_trajectory_processed


def _compute_resume_param_version(global_step: int, trigger_parameter_sync_step: int) -> int:
    if global_step <= 0:
        return 0
    return math.ceil(global_step / trigger_parameter_sync_step)


def _use_fully_async_mode(args) -> bool:
    return bool(
        getattr(args, "async_streaming", False)
        or getattr(args, "partial_rollout", False)
        or getattr(args, "staleness_threshold", 0.0) > 0
        or getattr(args, "trigger_parameter_sync_step", 1) > 1
    )


def _count_prompt_groups(num_trajectories: int, n_samples_per_prompt: int) -> int:
    if n_samples_per_prompt <= 0:
        raise ValueError(f"n_samples_per_prompt must be positive, got {n_samples_per_prompt}")
    return math.ceil(num_trajectories / n_samples_per_prompt)


@ray.remote(num_cpus=0, max_concurrency=4)
class VLLMLock:
    """跨 Actor 的互斥锁，保护 vLLM 临界区。

    确保生成和权重广播不会同时在同一个 vLLM 引擎上进行，
    保证同一批次中的所有样本使用一致的权重。
    """

    def __init__(self):
        import asyncio

        self._lock = asyncio.Lock()

    async def acquire(self):
        await self._lock.acquire()

    async def release(self):
        self._lock.release()


@ray.remote(max_concurrency=2)
class GenerateSamplesActor:
    """异步样本生成 Actor，支持流式生成和 staleness 控制。

    max_concurrency=2 允许 reset_staleness 在 fit 运行期间被调用，
    避免同步 Ray Actor 的单线程死锁。

    参考 verl 的 FullyAsyncRollouter 设计：
    - 支持 sample-level 流式生成（单样本完成即入队）
    - 支持 staleness_threshold 过期样本控制
    - 支持参数版本追踪（param_version）
    - 支持 partial_rollout（pause/resume vLLM）
    """

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
        self._partial_rollout = getattr(strategy.args, "partial_rollout", False)
        self._streaming = getattr(strategy.args, "async_streaming", False)
        self._full_async_mode = _use_fully_async_mode(strategy.args)
        self.rollout_queue = rollout_queue
        self.rollout_slots = rollout_slots
        self._last_eval_step = -1
        self._eval_just_done = False
        self._latest_global_step = 0

        # ==================== 全异步参数（参考 verl） ====================
        self._staleness_threshold = getattr(strategy.args, "staleness_threshold", 0.0)
        self._trigger_parameter_sync_step = getattr(strategy.args, "trigger_parameter_sync_step", 1)
        self._current_param_version = 0
        self._staleness_samples = 0
        self._required_samples = self.args.rollout_batch_size
        self._max_staleness_samples = int(
            self._required_samples
            * (self._staleness_threshold + 1)
            * self._trigger_parameter_sync_step
        )

        # threading.Condition 用于 staleness 暂停/恢复，
        # 解决原先 time.sleep 忙轮询 + 单线程 Actor 死锁问题。
        self._staleness_lock = threading.Lock()
        self._staleness_condition = threading.Condition(self._staleness_lock)

        self._total_generated_samples = 0
        self._step_start_time = time.time()
        self._idle_time_accumulated = 0.0

        logger.info(
            f"[GenerateSamplesActor] staleness_threshold={self._staleness_threshold}, "
            f"trigger_parameter_sync_step={self._trigger_parameter_sync_step}, "
            f"required_samples={self._required_samples}, "
            f"max_staleness_samples={self._max_staleness_samples}, "
            f"streaming={self._streaming}, partial_rollout={self._partial_rollout}, "
            f"full_async_mode={self._full_async_mode}"
        )

    def get_max_steps(self):
        return self.max_steps

    def load_state_dict(self, state_dict):
        self.prompts_dataloader.load_state_dict(state_dict)

    def set_param_version(self, param_version: int):
        self._current_param_version = param_version
        self._staleness_samples = 0

    def _refresh_latest_global_step(self) -> int:
        latest_global_step = self._latest_global_step
        while True:
            try:
                latest_global_step = self.rollout_slots.get(block=False)
            except Empty:
                break
        self._latest_global_step = latest_global_step
        return latest_global_step

    def _should_eval(self, global_step):
        return (
            self.eval_dataloader is not None
            and self.args.eval_steps != float("inf")
            and global_step > 0
            and global_step % self.args.eval_steps == 0
            and global_step != self._last_eval_step
        )

    def _run_eval(self):
        """执行评估并返回指标字典。"""
        logger.info("Starting async evaluation...")
        eval_kwargs = self.generate_kwargs.copy()
        eval_kwargs["temperature"] = self.args.eval_temperature
        eval_kwargs["n_samples_per_prompt"] = self.args.eval_n_samples_per_prompt

        samples_list = self.samples_generator.generate_eval_samples(**eval_kwargs)
        logs = compute_eval_metrics(self.eval_dataloader, samples_list, self.args.eval_n_samples_per_prompt)
        logger.info(f"Async evaluation completed: {logs}")
        return logs

    def _should_pause_generation(self) -> bool:
        """判断是否应暂停生成（参考 verl 的 _should_pause_generation）。

        当 staleness_threshold > 0 时，如果已生成的样本数超过 max_staleness_samples，
        则暂停生成，等待 Trainer 消费并触发参数同步。
        """
        if self._staleness_threshold <= 0:
            return False
        return self._staleness_samples >= self._max_staleness_samples

    def reset_staleness(self, new_param_version: int):
        """Trainer 同步权重后调用，重置 staleness 计数器并唤醒暂停的 fit 循环。

        通过 threading.Condition.notify_all() 唤醒因 staleness 暂停的生成循环，
        替代原先的 time.sleep 忙轮询。

        Returns:
            dict: 包含 idle_ratio 等监控指标
        """
        with self._staleness_lock:
            old_staleness = self._staleness_samples
            self._staleness_samples = 0
            self._current_param_version = new_param_version
            self._staleness_condition.notify_all()

        now = time.time()
        version_time = now - self._step_start_time
        idle_time = self._idle_time_accumulated
        active_time = version_time - idle_time
        idle_ratio = idle_time / version_time if version_time > 0 else 0

        timing_raw = {
            "fully_async/rollouter/active_time": active_time,
            "fully_async/rollouter/version_time": version_time,
            "fully_async/rollouter/idle_ratio": idle_ratio,
            "fully_async/rollouter/staleness_samples_at_sync": old_staleness,
        }
        logger.info(
            f"[GenerateSamplesActor] reset_staleness: param_version={new_param_version}, "
            f"old_staleness={old_staleness}, idle_ratio={idle_ratio:.4f}"
        )
        self._step_start_time = now
        self._idle_time_accumulated = 0.0
        return timing_raw

    def fit(self, start_episode: int, total_consumed_prompts: int) -> None:
        """入口方法：确保无论正常完成还是异常退出都发送 done 哨兵，防止 Trainer 死锁。"""
        try:
            self._fit_body(start_episode, total_consumed_prompts)
        except Exception:
            logger.exception("[GenerateSamplesActor] fit() crashed")
            raise
        finally:
            try:
                self.rollout_queue.put("done", block=True, timeout=30)
            except Exception:
                logger.warning("[GenerateSamplesActor] Failed to send 'done' sentinel to rollout_queue")

    def _fit_body(self, start_episode: int, total_consumed_prompts: int) -> None:
        if self._full_async_mode:
            self._fit_fully_async(start_episode, total_consumed_prompts)
            return

        for episode in range(start_episode, self.args.num_episodes):
            dataset_length = len(self.prompts_dataloader)
            pbar = tqdm(
                range(dataset_length),
                desc=f"Episode [{episode + 1}/{self.args.num_episodes}]",
                initial=total_consumed_prompts % max(dataset_length, 1),
            )
            while True:
                global_step = self.rollout_slots.get(block=True)

                # staleness 暂停：归还 slot 后通过 Condition.wait() 高效挂起，
                # 由 reset_staleness 的 notify_all() 唤醒。
                if self._should_pause_generation():
                    logger.info(
                        f"[GenerateSamplesActor] Paused: staleness_samples={self._staleness_samples} "
                        f">= max={self._max_staleness_samples}. Waiting for reset_staleness..."
                    )
                    self.rollout_slots.put(global_step, block=True)
                    pause_start = time.time()
                    with self._staleness_lock:
                        while self._should_pause_generation():
                            self._staleness_condition.wait()
                    self._idle_time_accumulated += time.time() - pause_start
                    continue

                # 评估：仅用 VLLMLock 互斥（不 pause 引擎，避免死锁）。
                # 原实现在 partial_rollout 模式下先 pause 再 eval，
                # 但 eval 需要向已暂停的引擎派发请求，导致死锁。
                if self._should_eval(global_step) and not self._eval_just_done:
                    self._eval_just_done = True
                    self._last_eval_step = global_step
                    ray.get(self.vllm_lock.acquire.remote())
                    try:
                        eval_metrics = self._run_eval()
                    finally:
                        ray.get(self.vllm_lock.release.remote())
                    self.rollout_queue.put(("eval", global_step, eval_metrics), block=True)
                    continue
                self._eval_just_done = False

                if not self._partial_rollout:
                    ray.get(self.vllm_lock.acquire.remote())
                try:
                    t0 = time.time()
                    if self._streaming or self._partial_rollout:
                        rollout_samples, filter_pass_rate, prompts_consumed, is_exhausted = (
                            self.samples_generator.generate_samples_fully_async(**self.generate_kwargs)
                        )
                    else:
                        rollout_samples, filter_pass_rate, prompts_consumed, is_exhausted = (
                            self.samples_generator.generate_samples(**self.generate_kwargs)
                        )
                    generation_time = time.time() - t0
                    total_consumed_prompts += prompts_consumed
                finally:
                    if not self._partial_rollout:
                        ray.get(self.vllm_lock.release.remote())

                produced = bool(rollout_samples)
                if produced:
                    param_version = self._current_param_version
                    with self._staleness_lock:
                        self._staleness_samples += _count_prompt_groups(
                            len(rollout_samples),
                            self.args.n_samples_per_prompt,
                        )
                    self._total_generated_samples += len(rollout_samples)

                    client_states = {
                        "episode": episode,
                        "total_consumed_prompts": total_consumed_prompts,
                        "data_loader_state_dict": self.prompts_dataloader.state_dict(),
                    }
                    self.rollout_queue.put(
                        (rollout_samples, client_states, filter_pass_rate, generation_time, param_version),
                        block=True,
                    )
                    if prompts_consumed:
                        pbar.update(prompts_consumed)
                else:
                    self.rollout_slots.put(global_step, block=True)

                if is_exhausted:
                    break

            pbar.close()

    def _fit_fully_async(self, start_episode: int, total_consumed_prompts: int) -> None:
        for episode in range(start_episode, self.args.num_episodes):
            dataset_length = len(self.prompts_dataloader)
            pbar = tqdm(
                range(dataset_length),
                desc=f"Episode [{episode + 1}/{self.args.num_episodes}]",
                initial=total_consumed_prompts % max(dataset_length, 1),
            )
            while True:
                global_step = self._refresh_latest_global_step()

                if self._should_pause_generation():
                    logger.info(
                        f"[GenerateSamplesActor] Paused: staleness_samples={self._staleness_samples} "
                        f">= max={self._max_staleness_samples}. Waiting for reset_staleness..."
                    )
                    pause_start = time.time()
                    with self._staleness_lock:
                        while self._should_pause_generation():
                            self._staleness_condition.wait()
                    self._idle_time_accumulated += time.time() - pause_start
                    continue

                if self._should_eval(global_step) and not self._eval_just_done:
                    self._eval_just_done = True
                    self._last_eval_step = global_step
                    ray.get(self.vllm_lock.acquire.remote())
                    try:
                        eval_metrics = self._run_eval()
                    finally:
                        ray.get(self.vllm_lock.release.remote())
                    self.rollout_queue.put(("eval", global_step, eval_metrics), block=True)
                    continue
                self._eval_just_done = False

                if not self._partial_rollout:
                    ray.get(self.vllm_lock.acquire.remote())
                try:
                    t0 = time.time()
                    prompt_group, prompts_consumed, is_exhausted = self.samples_generator.stream_prompt_group_fully_async(
                        **self.generate_kwargs
                    )
                    generation_time = time.time() - t0
                    total_consumed_prompts += prompts_consumed
                finally:
                    if not self._partial_rollout:
                        ray.get(self.vllm_lock.release.remote())

                if prompt_group is not None:
                    with self._staleness_lock:
                        self._staleness_samples += 1
                    self._total_generated_samples += len(prompt_group)
                    client_states = {
                        "episode": episode,
                        "total_consumed_prompts": total_consumed_prompts,
                        "data_loader_state_dict": self.prompts_dataloader.state_dict(),
                    }
                    self.rollout_queue.put(
                        ("sample", prompt_group, client_states, generation_time, self._current_param_version, prompts_consumed),
                        block=True,
                    )
                    if prompts_consumed:
                        pbar.update(prompts_consumed)

                if is_exhausted:
                    break

            pbar.close()


@ray.remote
class TrainingActor(BasePPOTrainer):
    """异步训练 Actor，支持多步本地训练和参数版本管理。

    参考 verl 的 FullyAsyncTrainer 设计：
    - 支持 trigger_parameter_sync_step（多步本地训练后才同步权重）
    - 支持参数版本追踪（param_version）
    - 支持 staleness 统计指标
    - 支持 MetricsAggregator 跨步指标聚合
    """

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
        generator_actor=None,
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
        self._partial_rollout = getattr(strategy.args, "partial_rollout", False)
        self._full_async_mode = _use_fully_async_mode(strategy.args)
        self.rollout_queue = rollout_queue
        self.rollout_slots = rollout_slots
        self.generator_actor = generator_actor

        # ==================== 全异步参数（参考 verl） ====================
        self._trigger_parameter_sync_step = getattr(strategy.args, "trigger_parameter_sync_step", 1)
        # 当前本地训练步数（1 到 trigger_parameter_sync_step 之间循环）
        self._local_trigger_step = 1
        # 当前参数版本号（每次同步权重后递增）
        self._current_param_version = 0
        # 过期样本统计
        self._stale_trajectory_processed = 0
        # 聚合指标
        self._metrics_aggregator = defaultdict(list)

        logger.info(
            f"[TrainingActor] trigger_parameter_sync_step={self._trigger_parameter_sync_step}"
        )

    def _publish_global_step(self, global_step: int) -> None:
        """Publish the latest global_step to the generator via rollout_slots.

        rollout_slots is a maxsize=1 Ray Queue used as a "latest value" channel.
        The generator reads it non-blockingly; if it misses a write it simply
        keeps its cached value, so a benign race here is acceptable.
        """
        if self.rollout_slots is None or not self._full_async_mode:
            return
        try:
            self.rollout_slots.get(block=False)
        except Empty:
            pass
        self.rollout_slots.put(global_step, block=True)

    def restore_async_state(self, global_step: int, synced_weights: bool = False):
        if global_step < 0:
            raise ValueError(f"global_step must be non-negative, got {global_step}")

        if synced_weights and global_step > 0:
            self._current_param_version = _compute_resume_param_version(
                global_step,
                self._trigger_parameter_sync_step,
            )
            self._local_trigger_step = 1
        else:
            self._current_param_version = global_step // self._trigger_parameter_sync_step
            self._local_trigger_step = global_step % self._trigger_parameter_sync_step + 1

    def fit(self, global_step: int = 0) -> None:
        """入口方法：统一异常处理和 logger 关闭。"""
        try:
            self._fit_body(global_step)
        except Exception:
            logger.exception("[TrainingActor] fit() crashed")
            raise
        finally:
            if self.wandb_logger:
                self.wandb_logger.close()
            if self.tensorboard_logger:
                self.tensorboard_logger.close()

    def _fit_body(self, global_step: int = 0) -> None:
        if self._full_async_mode:
            self._fit_fully_async(global_step)
            return

        step_start_time = time.time()
        self._latest_client_states = {}
        while True:
            payload = self.rollout_queue.get(block=True)
            if payload == "done":
                break

            # 处理来自 Generator 的评估结果
            if payload[0] == "eval":
                _, eval_step, eval_metrics = payload
                self.rollout_slots.put(global_step, block=True)
                logger.info(f"Eval at step {eval_step}: {eval_metrics}")
                if self.wandb_logger:
                    self.wandb_logger.log_eval(eval_step, eval_metrics)
                if self.tensorboard_logger:
                    self.tensorboard_logger.log_eval(eval_step, eval_metrics)
                # Save best checkpoint if this eval metric is the best so far
                client_states = dict(self._latest_client_states)
                client_states["global_step"] = global_step
                self.save_best_checkpoint(eval_metrics, eval_step, client_states)
                # Reset so the next training step's timing excludes eval overhead.
                step_start_time = time.time()
                continue

            rollout_samples, client_states, filter_pass_rate, generation_time, param_version = payload

            # 归还 slot token，允许 Generator 生产下一批
            self.rollout_slots.put(global_step, block=True)

            # 收集 staleness 统计（参考 verl 的 _collect_metrics_from_samples）
            staleness_info = self._collect_staleness_metrics(rollout_samples, param_version)

            # 执行 PPO 训练步骤
            status, global_step = self.train_step(rollout_samples, global_step)

            status["timing/generation"] = generation_time
            status["timing/step_total"] = time.time() - step_start_time
            step_start_time = time.time()

            if self.args.dynamic_filtering:
                status["dynamic_filtering_pass_rate"] = filter_pass_rate

            # 合并 staleness 指标
            status.update(staleness_info)

            # 更新本地步数和决定是否同步权重
            self._update_local_step()

            # 聚合指标到 metrics_aggregator
            self._aggregate_metrics(status)

            log_status = {k: v for k, v in status.items() if k not in ["generated_samples"]}
            logger.info(f"Global step {global_step}: {log_status}")

            client_states.update({"global_step": global_step})
            self._latest_client_states = client_states
            self.save_logs_and_checkpoints(global_step, status, client_states)

        # 训练结束前，如果还有未同步的步数，做最后一次同步
        if self._local_trigger_step > 1:
            self._do_weight_sync()

    def _process_queue_payload(self, payload, buffered_prompt_groups, global_step):
        """Process one item from rollout_queue.

        Returns ``(is_done, eval_handled)`` where *is_done* means the "done"
        sentinel was received and *eval_handled* means an eval result was logged.
        """
        if payload == "done":
            return True, False

        if payload[0] == "eval":
            _, eval_step, eval_metrics = payload
            logger.info(f"Eval at step {eval_step}: {eval_metrics}")
            if self.wandb_logger:
                self.wandb_logger.log_eval(eval_step, eval_metrics)
            if self.tensorboard_logger:
                self.tensorboard_logger.log_eval(eval_step, eval_metrics)
            client_states = dict(self._latest_client_states)
            client_states["global_step"] = global_step
            self.save_best_checkpoint(eval_metrics, eval_step, client_states)
            return False, True

        _, prompt_group, client_states, generation_time, param_version, prompts_consumed = payload
        buffered_prompt_groups.append(
            {
                "prompt_group": prompt_group,
                "client_states": client_states,
                "generation_time": generation_time,
                "param_version": param_version,
                "prompts_consumed": prompts_consumed,
            }
        )
        self._latest_client_states = client_states
        return False, False

    def _fit_fully_async(self, global_step: int = 0) -> None:
        step_start_time = time.time()
        self._latest_client_states = {}
        buffered_prompt_groups = deque()
        stream_exhausted = False

        self._publish_global_step(global_step)
        while True:
            target_prompt_groups = self.args.rollout_batch_size
            should_train = len(buffered_prompt_groups) >= target_prompt_groups
            if stream_exhausted and buffered_prompt_groups:
                should_train = True

            if not should_train:
                payload = self.rollout_queue.get(block=True)
                is_done, eval_handled = self._process_queue_payload(
                    payload, buffered_prompt_groups, global_step,
                )
                if is_done:
                    stream_exhausted = True
                    if not buffered_prompt_groups:
                        break
                if eval_handled:
                    step_start_time = time.time()
                continue

            # Non-blocking drain: process any pending eval/sample messages
            # so that eval results are not delayed behind training steps.
            while True:
                try:
                    payload = self.rollout_queue.get(block=False)
                except Empty:
                    break
                is_done, eval_handled = self._process_queue_payload(
                    payload, buffered_prompt_groups, global_step,
                )
                if is_done:
                    stream_exhausted = True
                if eval_handled:
                    step_start_time = time.time()

            prompt_group_batch_size = min(len(buffered_prompt_groups), target_prompt_groups)
            if prompt_group_batch_size < target_prompt_groups:
                logger.warning(
                    f"[TrainingActor] Partial batch: {prompt_group_batch_size}/{target_prompt_groups} prompt groups "
                    f"(stream_exhausted={stream_exhausted}). Training may produce suboptimal gradients."
                )
            batch_items = [buffered_prompt_groups.popleft() for _ in range(prompt_group_batch_size)]
            rollout_samples = []
            total_generation_time = 0.0
            total_prompts_consumed = 0
            fallback_param_version = 0

            for item in batch_items:
                fallback_param_version = max(fallback_param_version, item["param_version"])
                total_generation_time += item["generation_time"]
                total_prompts_consumed += item["prompts_consumed"]
                for sample in item["prompt_group"]:
                    sample_info = sample.info if sample.info is not None else {}
                    sample_info.setdefault("min_weight_version", item["param_version"])
                    sample_info.setdefault("max_weight_version", item["param_version"])
                    sample_info.setdefault("partial_old_token_count", 0)
                    sample.info = sample_info
                    rollout_samples.append(sample)

            client_states = dict(batch_items[-1]["client_states"])
            staleness_info = self._collect_staleness_metrics(rollout_samples, fallback_param_version)
            status, global_step = self.train_step(rollout_samples, global_step)

            status["timing/generation"] = total_generation_time
            status["timing/step_total"] = time.time() - step_start_time
            step_start_time = time.time()

            if self.args.dynamic_filtering and total_prompts_consumed:
                status["dynamic_filtering_pass_rate"] = prompt_group_batch_size / total_prompts_consumed * 100

            status.update(staleness_info)
            self._update_local_step()
            self._aggregate_metrics(status)

            log_status = {k: v for k, v in status.items() if k not in ["generated_samples"]}
            logger.info(f"Global step {global_step}: {log_status}")

            client_states.update({"global_step": global_step})
            self._latest_client_states = client_states
            self.save_logs_and_checkpoints(global_step, status, client_states)
            self._publish_global_step(global_step)

            if stream_exhausted and not buffered_prompt_groups:
                break

        if self._local_trigger_step > 1:
            self._do_weight_sync()

    def train_step(self, rollout_samples, global_step: int):
        """重写 BasePPOTrainer.train_step，跳过其中的 broadcast_to_vllm 调用。

        权重同步由 _update_local_step 根据 trigger_parameter_sync_step 统一控制，
        而不是每个 train_step 都同步一次。这是参考 verl FullyAsyncTrainer 的核心优化。
        """
        import transformers
        _TRANSFORMERS_V5 = int(transformers.__version__.split(".")[0]) >= 5

        from openrlhf.trainer.ppo_utils.replay_buffer import balance_experiences

        # 将 rollout 样本转为 PPO 训练数据
        t0 = time.time()
        experiences = self.experience_maker.make_experience_batch(rollout_samples)
        make_experience_time = time.time() - t0

        # 快速检查第一个解码样本
        _decode = self.tokenizer.decode if _TRANSFORMERS_V5 else self.tokenizer.batch_decode
        sample0 = [
            _decode(experiences[0].sequences[0].unsqueeze(0), skip_special_tokens=True)[0],
            experiences[0].info["reward"][0].item(),
        ]
        print(sample0)

        # 按需均衡 experience
        if self.args.use_dynamic_batch:
            experiences = balance_experiences(experiences, self.args)

        # 推送 experience 到 actor/critic 分片
        refs = self.actor_model_group.async_run_method_batch(method_name="append", experience=experiences)
        if self.critic_model_group is not None:
            refs.extend(self.critic_model_group.async_run_method_batch(method_name="append", experience=experiences))
        ray.get(refs)

        # 执行 PPO 优化
        t0 = time.time()
        status = self.ppo_train(global_step)
        ppo_train_time = time.time() - t0

        # 注意：此处**不再调用** broadcast_to_vllm()
        # 权重同步由 _update_local_step -> _do_weight_sync 统一控制

        # 更新 KL controller
        if "kl" in status:
            self.kl_ctl.update(status["kl"], len(rollout_samples))

        # 计时分解
        status["timing/make_experience"] = make_experience_time
        status["timing/ppo_train"] = ppo_train_time
        status["timing/broadcast"] = 0  # 由 _do_weight_sync 统一记录

        status["generated_samples"] = sample0
        return status, global_step + 1

    def _collect_staleness_metrics(self, rollout_samples, fallback_param_version: int) -> dict:
        """收集样本过期和 partial rollout 的统计指标。"""
        metrics, self._stale_trajectory_processed = _summarize_rollout_versions(
            rollout_samples,
            current_param_version=self._current_param_version,
            fallback_param_version=fallback_param_version,
            stale_trajectory_processed=self._stale_trajectory_processed,
        )
        metrics["fully_async/local_trigger_step"] = self._local_trigger_step
        return metrics

    def _update_local_step(self):
        """更新本地训练步数，判断是否触发参数同步（参考 verl 的 _fit_update_local_step）。

        每执行 trigger_parameter_sync_step 步本地训练后，同步权重到 vLLM 并递增参数版本。
        """
        logger.info(
            f"[TrainingActor] local_trigger_step={self._local_trigger_step}/"
            f"{self._trigger_parameter_sync_step}, param_version={self._current_param_version}"
        )

        if self._local_trigger_step < self._trigger_parameter_sync_step:
            self._local_trigger_step += 1
        else:
            # 达到同步步数，触发权重同步
            self._do_weight_sync()
            self._current_param_version += 1
            self._local_trigger_step = 1

    def _do_weight_sync(self):
        """执行权重同步到 vLLM，并通知 Generator 重置 staleness。"""
        t0 = time.time()
        if self.vllm_engines is not None:
            self.broadcast_to_vllm(target_weight_version=self._current_param_version + 1)
        sync_time = time.time() - t0
        logger.info(
            f"[TrainingActor] Weight sync completed in {sync_time:.4f}s, "
            f"new param_version={self._current_param_version + 1}"
        )

        # 通知 Generator 重置 staleness（参考 verl 的 reset_staleness）
        if self.generator_actor is not None:
            rollouter_timing = ray.get(
                self.generator_actor.reset_staleness.remote(self._current_param_version + 1)
            )
            # 将 rollouter 的 timing 信息记录到 aggregator
            for k, v in rollouter_timing.items():
                self._metrics_aggregator[k].append(v)

        # 输出聚合指标并重置
        aggregated = self._get_aggregated_metrics()
        if aggregated and self.wandb_logger:
            self.wandb_logger.log_train(self._current_param_version, aggregated)
        if aggregated and self.tensorboard_logger:
            self.tensorboard_logger.log_train(self._current_param_version, aggregated)
        self._metrics_aggregator.clear()

    def _aggregate_metrics(self, status: dict):
        """聚合多步训练的指标（参考 verl 的 MetricsAggregator）。"""
        for k, v in status.items():
            if isinstance(v, (int, float)):
                self._metrics_aggregator[k].append(v)

    def _get_aggregated_metrics(self) -> dict:
        """获取聚合后的指标。

        时间类指标求和，其他指标求平均，计数类指标取最后值。
        """
        if not self._metrics_aggregator:
            return {}

        result = {}
        for k, values in self._metrics_aggregator.items():
            if not values:
                continue
            if "timing/" in k or "time" in k.lower():
                # 时间类指标求和
                result[f"aggregated/{k}"] = sum(values)
            elif "/count/" in k or "param_version" in k:
                # 计数类指标取最后值
                result[f"aggregated/{k}"] = values[-1]
            elif "max_partial_span" in k or "max_version_gap" in k:
                result[f"aggregated/{k}"] = max(values)
            else:
                # 其他指标求平均
                result[f"aggregated/{k}"] = sum(values) / len(values)
        return result

    def broadcast_to_vllm(self, target_weight_version: int | None = None):
        """广播权重到 vLLM（加锁 + pause/resume 保护）。

        partial_rollout 模式下使用 pause_generation(mode="abort") 中止正在进行的请求，
        生成侧会保留已完成的旧 token 并在新权重下续生成剩余部分。
        """
        if target_weight_version is None:
            target_weight_version = self._current_param_version + 1
        ray.get(self.vllm_lock.acquire.remote())
        if self._partial_rollout:
            batch_vllm_engine_call(self.vllm_engines, "pause_generation", mode="abort")
        try:
            super().broadcast_to_vllm()
            batch_vllm_engine_call(self.vllm_engines, "set_weight_version", weight_version=target_weight_version)
        finally:
            if self._partial_rollout:
                batch_vllm_engine_call(self.vllm_engines, "resume_generation")
            ray.get(self.vllm_lock.release.remote())


@ray.remote
class PPOTrainerAsync:
    """全异步 PPO 训练编排器。

    参考 verl 的 FullyAsyncPolicy 架构：
    - GenerateSamplesActor 持续生成样本，通过 Queue 传输
    - TrainingActor 持续消费样本进行训练
    - 支持 staleness_threshold 控制 off-policy 程度
    - 支持 trigger_parameter_sync_step 多步本地训练
    - 支持参数版本追踪和流式生成
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
        self.strategy = strategy
        self._full_async_mode = _use_fully_async_mode(strategy.args)
        # 设置 eval 和 save 步数
        if strategy.args.eval_steps == -1:
            strategy.args.eval_steps = float("inf")
        if strategy.args.save_steps == -1:
            strategy.args.save_steps = float("inf")

        queue_size = getattr(strategy.args, "async_queue_size", 1)
        if queue_size <= 0:
            raise ValueError(f"async_queue_size must be positive, got {queue_size}")

        # 如果使用 staleness 控制，增大默认 queue_size
        staleness_threshold = getattr(strategy.args, "staleness_threshold", 0.0)
        trigger_sync_step = getattr(strategy.args, "trigger_parameter_sync_step", 1)
        required_prompt_groups = strategy.args.rollout_batch_size
        if self._full_async_mode:
            min_full_async_queue_size = max(
                1,
                int(required_prompt_groups * max(1.0, (staleness_threshold + 1) * trigger_sync_step)),
            )
            if queue_size < min_full_async_queue_size:
                logger.info(
                    f"[PPOTrainerAsync] Auto-adjusting queue_size from {queue_size} to {min_full_async_queue_size} "
                    f"for fully async mode"
                )
                queue_size = min_full_async_queue_size

        if staleness_threshold > 0 and queue_size == 1:
            # 自动调整 queue_size 以匹配 staleness 设置
            auto_queue_size = max(2, int((staleness_threshold + 1) * trigger_sync_step))
            logger.info(
                f"[PPOTrainerAsync] Auto-adjusting queue_size from {queue_size} to {auto_queue_size} "
                f"based on staleness_threshold={staleness_threshold}, trigger_sync_step={trigger_sync_step}"
            )
            queue_size = auto_queue_size

        if self._full_async_mode:
            if staleness_threshold <= 0:
                logger.warning(
                    "[PPOTrainerAsync] Full async mode with staleness_threshold=0: generation "
                    "is only throttled by rollout_queue capacity (maxsize=%d). Consider setting "
                    "staleness_threshold > 0 for better off-policy control.",
                    queue_size,
                )
            if not getattr(strategy.args, "partial_rollout", False):
                logger.warning(
                    "[PPOTrainerAsync] Full async mode without partial_rollout: vllm_lock is "
                    "acquired/released for every prompt group, causing high RPC overhead. "
                    "Consider enabling --partial_rollout for better throughput.",
                )

        logger.info(
            f"[PPOTrainerAsync] queue_size={queue_size}, staleness_threshold={staleness_threshold}, "
            f"trigger_parameter_sync_step={trigger_sync_step}, full_async_mode={self._full_async_mode}"
        )

        self.rollout_queue = Queue(maxsize=queue_size)

        # Old async mode uses rollout_slots as a counting semaphore.
        # Fully async mode uses a size-1 queue as a latest-global-step signal bus.
        if self._full_async_mode:
            self.rollout_slots = Queue(maxsize=1)
            self.rollout_slots.put(0, block=True)
        else:
            self.rollout_slots = Queue(maxsize=queue_size)
            for _ in range(queue_size):
                self.rollout_slots.put(0, block=True)

        # vLLM 互斥锁
        vllm_lock = VLLMLock.remote()

        self.generator_actor = GenerateSamplesActor.remote(
            pretrain=pretrain,
            strategy=strategy,
            vllm_engines=vllm_engines,
            vllm_lock=vllm_lock,
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
            vllm_lock=vllm_lock,
            rollout_queue=self.rollout_queue,
            rollout_slots=self.rollout_slots,
            generator_actor=self.generator_actor,
        )

    def fit(self) -> None:
        checkpoint_states = ray.get(self.trainer_actor.init_checkpoint_states.remote())
        ray.get(self.trainer_actor.restore_best_checkpoint_state.remote(checkpoint_states))

        # 恢复步数和 epoch
        start_episode = checkpoint_states["episode"]
        global_step = checkpoint_states["global_step"]
        total_consumed_prompts = checkpoint_states.get("total_consumed_prompts", 0)
        resume_param_version = _compute_resume_param_version(global_step, self.strategy.args.trigger_parameter_sync_step)
        ray.get(
            [
                self.trainer_actor.restore_async_state.remote(global_step, synced_weights=global_step > 0),
                self.generator_actor.set_param_version.remote(resume_param_version),
            ]
        )

        # 恢复时同步 vLLM 权重和 dataloader 状态
        if global_step > 0:
            ray.get(
                [
                    self.generator_actor.load_state_dict.remote(checkpoint_states["data_loader_state_dict"]),
                    self.trainer_actor.broadcast_to_vllm.remote(target_weight_version=resume_param_version),
                ]
            )

        # 启动异步训练（ray.wait 模式：任一 Actor 崩溃即终止对端，防止死锁）
        generator_future = self.generator_actor.fit.remote(
            start_episode=start_episode, total_consumed_prompts=total_consumed_prompts,
        )
        trainer_future = self.trainer_actor.fit.remote(global_step=global_step)

        actor_names = {
            generator_future: "GenerateSamplesActor",
            trainer_future: "TrainingActor",
        }
        peer_actors = {
            generator_future: self.trainer_actor,
            trainer_future: self.generator_actor,
        }
        futures = [generator_future, trainer_future]

        try:
            while futures:
                done, futures = ray.wait(futures, num_returns=1, timeout=None)
                for future in done:
                    try:
                        ray.get(future)
                        logger.info(f"[PPOTrainerAsync] {actor_names[future]} completed")
                    except Exception as e:
                        failed_name = actor_names[future]
                        logger.error(f"[PPOTrainerAsync] {failed_name} crashed: {e}")
                        for remaining in futures:
                            ray.cancel(remaining, force=True)
                        try:
                            ray.kill(peer_actors[future], no_restart=True)
                        except Exception:
                            pass
                        raise
        except Exception:
            for f in futures:
                ray.cancel(f, force=True)
            raise

    def get_max_steps(self):
        return ray.get(self.generator_actor.get_max_steps.remote())
