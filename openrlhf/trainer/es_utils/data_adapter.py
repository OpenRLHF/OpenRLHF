from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

from openrlhf.datasets import PromptDataset
from openrlhf.datasets.utils import blending_datasets

STABILIZE_SEED = -1
EVAL_SEED = -2


@dataclass
class ESExperience:
    """Minimal experience container for ES rollouts."""

    sequences: torch.Tensor = None
    attention_mask: torch.LongTensor = None
    action_mask: torch.BoolTensor = None
    rollout_log_probs: torch.Tensor = None
    seeds: torch.Tensor = None
    rewards: torch.Tensor = None
    prompts: List[str] = None
    labels: List[str] = None
    info: Dict[str, torch.Tensor] = None


@dataclass
class ESEvalSample:
    """Adapter for the shared PPO eval metric helper."""

    prompts: List[str]
    rewards: float
    response_length: Optional[torch.Tensor] = None
    truncated: Optional[torch.Tensor] = None


def _summarize_metric_series(prefix: str, values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        # Heuristic inactive on every sample: log zeros so TensorBoard stays defined.
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_count": 0.0,
        }
    return {
        f"{prefix}_mean": float(finite.mean()),
        f"{prefix}_min": float(finite.min()),
        f"{prefix}_max": float(finite.max()),
        f"{prefix}_std": float(finite.std()),
        f"{prefix}_count": float(finite.size),
    }


def summarize_experience_metrics(samples: List[ESExperience], prefix: str) -> Dict[str, float]:
    rewards: List[float] = []
    extra_logs: Dict[str, List[float]] = defaultdict(list)

    for sample in samples:
        if sample.rewards is not None:
            rewards.append(sample.rewards[0].item())
        for key, value in (sample.info or {}).items():
            extra_logs[key].append(value[0].item())

    metrics = _summarize_metric_series(f"{prefix}_reward", rewards)
    for key, values in extra_logs.items():
        metrics.update(_summarize_metric_series(f"{prefix}_{key}", values))
    return metrics


def prepare_datasets(strategy, tokenizer):
    """Prepare ES dataloaders using the shared OpenRLHF dataset flow."""
    args = strategy.args
    max_train = args.max_samples

    train_data = blending_datasets(
        args.prompt_data,
        args.prompt_data_probs,
        strategy,
        args.seed,
        max_count=max_train,
        dataset_split=args.prompt_split,
    )
    train_data = train_data.select(range(min(max_train, len(train_data))))
    prompts_dataset = PromptDataset(train_data, tokenizer, strategy, input_template=args.input_template)
    prompts_dataloader = strategy.setup_dataloader(prompts_dataset, 1, True, True, prompts_dataset.collate_fn)

    if args.eval_dataset:
        max_eval_samples = args.max_eval_samples if args.max_eval_samples is not None else max_train
        eval_data = blending_datasets(
            args.eval_dataset,
            args.eval_dataset_probs,
            strategy,
            args.seed,
            max_count=max_eval_samples,
            dataset_split=args.eval_split,
        )
        eval_data = eval_data.select(range(min(max_eval_samples, len(eval_data))))
        eval_dataset = PromptDataset(eval_data, tokenizer, strategy, input_template=args.input_template)
        eval_dataloader = strategy.setup_dataloader(eval_dataset, 1, True, False, eval_dataset.collate_fn)
    else:
        eval_dataloader = None

    num_prompts = len(prompts_dataset)
    rollout_batch_size = max(1, args.rollout_batch_size)
    if args.es_shared_batch:
        steps_per_epoch = max(1, (num_prompts + rollout_batch_size - 1) // rollout_batch_size)
    else:
        prompts_per_step = rollout_batch_size * args.population_size
        steps_per_epoch = max(1, (num_prompts + prompts_per_step - 1) // prompts_per_step)

    return prompts_dataloader, eval_dataloader, steps_per_epoch
