import asyncio
from typing import Dict, List, Optional, Tuple

import ray
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from vllm import SamplingParams

from openrlhf.trainer.es_utils.data_adapter import EVAL_SEED, STABILIZE_SEED, ESExperience
from openrlhf.trainer.ppo_utils.samples_generator import _collect_prompt_batch
from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call


class ESSamplesGenerator:
    """ES sample generator with staged mutation and shared or unique batch modes."""

    def __init__(
        self,
        strategy,
        prompts_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        tokenizer,
        vllm_engines: List,
        reward_model_group=None,
    ):
        self.strategy = strategy
        self.args = strategy.args
        self.tokenizer = tokenizer
        self.vllm_engines = vllm_engines or []
        self.prompts_dataloader = prompts_dataloader
        self.eval_dataloader = eval_dataloader
        self.reward_model_group = reward_model_group

        self._dataloader_iter = None
        self._eval_dataloader_iter = None
        self._cached_prompts = None
        self._cached_labels = None
        self._cached_exhausted = False
        self._cached_unique_seed_batches = None
        self._cached_unique_exhausted = False

    def reset_train_iterator(self) -> None:
        self._dataloader_iter = None
        self._cached_prompts = None
        self._cached_labels = None
        self._cached_exhausted = False
        self._cached_unique_seed_batches = None
        self._cached_unique_exhausted = False

    @torch.no_grad()
    def generate_samples(
        self,
        engine_seeds: List[int],
        es_std: float,
        shared_batch: bool = True,
        include_eval: bool = False,
        **generate_kwargs,
    ) -> Tuple[List[ESExperience], Optional[float], int, bool]:
        if self._dataloader_iter is None:
            self._dataloader_iter = iter(self.prompts_dataloader)

        self._cached_prompts = None
        self._cached_labels = None
        self._cached_exhausted = False
        self._cached_unique_seed_batches = None
        self._cached_unique_exhausted = False

        all_seeds = list(engine_seeds) + ([EVAL_SEED] if include_eval else [])
        experiences, exhausted, prompts_consumed = asyncio.run(
            self._run_all_seeds(
                all_seeds,
                es_std=es_std,
                shared_batch=shared_batch,
                generate_kwargs=generate_kwargs,
            )
        )
        return experiences, None, prompts_consumed, exhausted

    async def _run_all_seeds(
        self,
        engine_seeds: List[int],
        *,
        es_std: float,
        shared_batch: bool,
        generate_kwargs: Dict,
    ) -> Tuple[List[ESExperience], bool, int]:
        engine_pool: asyncio.Queue = asyncio.Queue()
        for engine in self.vllm_engines:
            engine_pool.put_nowait(engine)

        prompt_lock = asyncio.Lock()
        num_train_seeds = sum(seed != EVAL_SEED for seed in engine_seeds)
        tasks = []
        train_seed_idx = 0
        for seed in engine_seeds:
            current_train_seed_idx = None if seed == EVAL_SEED else train_seed_idx
            if seed != EVAL_SEED:
                train_seed_idx += 1
            tasks.append(
                asyncio.create_task(
                    self._run_one_seed(
                        train_seed_idx=current_train_seed_idx,
                        seed=seed,
                        engine_pool=engine_pool,
                        prompt_lock=prompt_lock,
                        es_std=es_std,
                        shared_batch=shared_batch,
                        num_train_seeds=num_train_seeds,
                        generate_kwargs=generate_kwargs,
                    )
                )
            )

        all_experiences: List[ESExperience] = []
        exhausted = False
        prompts_consumed = 0
        reward_wait_tasks: List[asyncio.Task] = []

        for task in asyncio.as_completed(tasks):
            round_experiences, batch_exhausted, consumed, reward_jobs, is_eval = await task
            all_experiences.extend(round_experiences)
            if not is_eval:
                prompts_consumed = consumed if shared_batch else prompts_consumed + consumed
                exhausted |= batch_exhausted
            for experiences, ref in reward_jobs:
                reward_wait_tasks.append(asyncio.create_task(self._await_and_apply_rewards(experiences, ref)))

        if reward_wait_tasks:
            await asyncio.gather(*reward_wait_tasks)

        return all_experiences, exhausted, prompts_consumed

    async def _await_and_apply_rewards(self, experiences: List[ESExperience], ref) -> None:
        reward_output = await asyncio.to_thread(ray.get, ref)
        self._apply_reward_output(experiences, reward_output)

    async def _run_one_seed(
        self,
        *,
        train_seed_idx: Optional[int],
        seed: int,
        engine_pool: asyncio.Queue,
        prompt_lock: asyncio.Lock,
        es_std: float,
        shared_batch: bool,
        num_train_seeds: int,
        generate_kwargs: Dict,
    ) -> Tuple[List[ESExperience], bool, int, List[Tuple[List[ESExperience], List]], bool]:
        is_eval = seed == EVAL_SEED
        engine = await engine_pool.get()
        try:
            if is_eval:
                await asyncio.to_thread(ray.get, engine.model_mutate.remote(None, 0.0))
                async with prompt_lock:
                    if self._eval_dataloader_iter is None:
                        self._eval_dataloader_iter = iter(self.eval_dataloader)
                    prompts, labels, _ = _collect_prompt_batch(
                        self._eval_dataloader_iter, len(self.eval_dataloader.dataset)
                    )
                    self._eval_dataloader_iter = None
                batch_exhausted = False
            else:
                mutate_seed = None if seed == STABILIZE_SEED else seed
                mutate_std = 0.0 if seed == STABILIZE_SEED else es_std
                await asyncio.to_thread(ray.get, engine.model_mutate.remote(mutate_seed, mutate_std))
                async with prompt_lock:
                    prompts, labels, batch_exhausted = self._get_prompts_for_seed(
                        train_seed_idx=train_seed_idx,
                        num_seeds=num_train_seeds,
                        num_prompts=self.args.rollout_batch_size,
                        shared_batch=shared_batch,
                    )

            round_experiences = await asyncio.to_thread(
                self._dispatch_and_collect,
                [(prompts, labels, seed)],
                [engine],
                **generate_kwargs,
            )
        finally:
            engine_pool.put_nowait(engine)

        reward_jobs = []
        if self.reward_model_group is not None and round_experiences:
            refs = self.reward_model_group.async_score(
                queries=self._decode_queries(round_experiences),
                prompts=[experience.prompts[0] if experience.prompts else "" for experience in round_experiences],
                labels=[experience.labels[0] if experience.labels else "" for experience in round_experiences],
            )
            reward_jobs.append((round_experiences, refs))

        return round_experiences, batch_exhausted, len(prompts), reward_jobs, is_eval

    def _get_prompts_for_seed(
        self,
        *,
        train_seed_idx: Optional[int],
        num_seeds: int,
        num_prompts: int,
        shared_batch: bool,
    ) -> Tuple[List[str], List[str], bool]:
        if shared_batch:
            if self._cached_prompts is None:
                self._cached_prompts, self._cached_labels, self._cached_exhausted = _collect_prompt_batch(
                    self._dataloader_iter, num_prompts
                )
            return self._cached_prompts, self._cached_labels, self._cached_exhausted

        if train_seed_idx is None:
            raise ValueError("train_seed_idx must be provided for non-eval seeds")

        if self._cached_unique_seed_batches is None:
            total_prompts = num_prompts * num_seeds
            prompts, labels, self._cached_unique_exhausted = _collect_prompt_batch(
                self._dataloader_iter, total_prompts
            )
            self._cached_unique_seed_batches = []
            for seed_idx in range(num_seeds):
                start = seed_idx * num_prompts
                end = start + num_prompts
                self._cached_unique_seed_batches.append((prompts[start:end], labels[start:end]))

        prompts, labels = self._cached_unique_seed_batches[train_seed_idx]
        exhausted = self._cached_unique_exhausted or len(prompts) < num_prompts
        return prompts, labels, exhausted

    def _dispatch_and_collect(
        self,
        prompts_per_engine: List[Tuple[List[str], List[str], int]],
        engines: List,
        **generate_kwargs,
    ) -> List[ESExperience]:
        stop = generate_kwargs.get("stop")
        if stop is None:
            token = generate_kwargs.get("stop_token")
            stop = [] if not token else [token]
        pm = generate_kwargs.get("prompt_max_len")
        mn = generate_kwargs.get("max_new_tokens")
        budget = generate_kwargs.get("max_len") or 2048
        unified = pm is None and mn is None
        sampling_params = SamplingParams(
            temperature=generate_kwargs.get("temperature", 1.0),
            top_p=generate_kwargs.get("top_p", 1.0),
            top_k=generate_kwargs.get("top_k", -1),
            max_tokens=None if unified else mn,
            min_tokens=generate_kwargs.get("min_new_tokens", 1),
            skip_special_tokens=generate_kwargs.get("skip_special_tokens", False),
            logprobs=None,
            stop=stop,
        )
        truncate_length = budget
        n_samples = self.args.n_samples_per_prompt

        all_refs = []
        total_samples = 0
        for engine_idx, (prompts, labels, seed) in enumerate(prompts_per_engine):
            ref = engines[engine_idx].generate_responses_batch.remote(
                prompts=prompts,
                labels=labels,
                sampling_params=sampling_params,
                max_length=truncate_length,
                hf_tokenizer=self.tokenizer,
                num_samples=n_samples,
            )
            all_refs.append((ref, seed))
            total_samples += len(prompts) * max(int(n_samples), 1)

        experiences: List[ESExperience] = []
        pbar = tqdm(total=total_samples, desc="Generate samples")
        pending = list(all_refs)
        while pending:
            ready_refs, _ = ray.wait([ref for ref, _ in pending], num_returns=1, timeout=10.0)
            for ready_ref in ready_refs:
                for index, (ref, seed) in enumerate(pending):
                    if ref != ready_ref:
                        continue
                    batch_responses = ray.get(ready_ref)
                    for response in batch_responses:
                        experiences.append(self._process_response_into_experience(response, seed, truncate_length))
                    pending.pop(index)
                    pbar.update(len(batch_responses))
                    break
        pbar.close()
        return experiences

    def _decode_queries(self, experiences: List[ESExperience]) -> List[str]:
        queries = []
        for experience in experiences:
            token_ids = experience.sequences.squeeze(0).tolist()
            queries.append(self.tokenizer.decode(token_ids, skip_special_tokens=False))
        return queries

    def _apply_reward_output(self, experiences: List[ESExperience], reward_output: Dict) -> None:
        rewards = reward_output.get("rewards", [])
        extra_logs = reward_output.get("extra_logs", {})
        for idx, experience in enumerate(experiences):
            if idx < len(rewards):
                experience.rewards = torch.tensor([float(rewards[idx])], dtype=torch.float32)
            for key, values in extra_logs.items():
                if idx < len(values):
                    v = values[idx]
                    x = float("nan") if v is None else float(v)
                    experience.info[key] = torch.tensor([x], dtype=torch.float32)

    def _process_response_into_experience(
        self,
        response: dict,
        seed: int,
        truncate_length: int,
    ) -> ESExperience:
        tokenized_observation = response["observation_tokens"].copy()
        tokenized_ranges = response["action_ranges"]
        # MultiTurnAgentExecutor stores float; SingleTurnAgentExecutor now normalises
        # to float at the source (see agent.py). None means no reward endpoint.
        reward_val: Optional[float] = response.get("reward")
        reward_tensor = None if reward_val is None else torch.tensor([reward_val], dtype=torch.float32)

        sequences = torch.tensor(tokenized_observation, dtype=torch.long)
        attention_mask = torch.ones(len(tokenized_observation), dtype=torch.long)
        action_mask = torch.zeros_like(attention_mask)
        for start, end in tokenized_ranges:
            action_mask[start:end] = 1

        sequences = sequences[:truncate_length].to("cpu")
        attention_mask = attention_mask[:truncate_length].to("cpu")
        action_mask = action_mask[1:truncate_length].to("cpu")

        rollout_log_probs = None
        if response.get("rollout_log_probs") is not None:
            rollout_log_probs = torch.tensor(response["rollout_log_probs"][1:truncate_length]).to("cpu")

        info = {}
        for key, value in (response.get("extra_logs") or {}).items():
            # agent.py normalises extra_log values to plain float scalars at the source
            x = float("nan") if value is None else float(value)
            info[key] = torch.tensor([x], dtype=torch.float32)

        return ESExperience(
            sequences=sequences.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
            action_mask=action_mask.unsqueeze(0),
            rollout_log_probs=rollout_log_probs.unsqueeze(0) if rollout_log_probs is not None else None,
            seeds=torch.tensor([seed]),
            rewards=reward_tensor,
            prompts=[response.get("prompt", "")],
            labels=[response.get("label", "")],
            info=info,
        )

    @torch.no_grad()
    def generate_eval_samples(self, **generate_kwargs) -> List[ESExperience]:
        if self._eval_dataloader_iter is None:
            self._eval_dataloader_iter = iter(self.eval_dataloader)

        try:
            if self.args.vllm_enable_sleep:
                batch_vllm_engine_call(self.vllm_engines, "wake_up")

            ray.get([engine.model_mutate.remote(None, 0.0) for engine in self.vllm_engines])
            prompts, labels, _ = _collect_prompt_batch(self._eval_dataloader_iter, len(self.eval_dataloader.dataset))

            num_engines = len(self.vllm_engines)
            prompts_per_engine = []
            prompts_per_engine_count = len(prompts) // num_engines + 1
            for index in range(num_engines):
                start = index * prompts_per_engine_count
                end = min(start + prompts_per_engine_count, len(prompts))
                if start < len(prompts):
                    prompts_per_engine.append((prompts[start:end], labels[start:end], 0))

            experiences = self._dispatch_and_collect(prompts_per_engine, self.vllm_engines, **generate_kwargs)
            if self.reward_model_group is not None and experiences:
                reward_ref = self.reward_model_group.async_score(
                    queries=self._decode_queries(experiences),
                    prompts=[experience.prompts[0] if experience.prompts else "" for experience in experiences],
                    labels=[experience.labels[0] if experience.labels else "" for experience in experiences],
                )
                self._apply_reward_output(experiences, ray.get(reward_ref))
            return experiences
        finally:
            if self.args.vllm_enable_sleep:
                batch_vllm_engine_call(self.vllm_engines, "sleep")
            self._eval_dataloader_iter = None
