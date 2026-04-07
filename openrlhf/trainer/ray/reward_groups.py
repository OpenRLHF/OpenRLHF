from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import ray
import torch

from openrlhf.reward.core import (
    NamedRewardComponent,
    instantiate_heuristic,
    load_heuristic_classes,
    merge_optional_values,
    sanitize_extra_logs,
)


@ray.remote
class HeuristicWorkerActor:
    """Ray worker that owns exactly one heuristic instance."""

    def __init__(self, heuristics_path: str, class_name: str):
        heuristic_cls = next(cls for cls in load_heuristic_classes(heuristics_path) if cls.__name__ == class_name)
        has_gpu = bool(ray.get_gpu_ids()) and torch.cuda.is_available()
        device = torch.device("cuda" if has_gpu else "cpu")
        self.component = NamedRewardComponent(
            heuristic_cls.__name__,
            instantiate_heuristic(heuristic_cls, device=device),
        )
        self.name = heuristic_cls.__name__

    def score_batch(
        self, queries: List[str], prompts: List[str], labels: List[str]
    ) -> Dict[str, List[Optional[float]]]:
        batch_size = len(prompts)
        indices = [idx for idx, label in enumerate(labels) if self.component.should_call(label)]
        if not indices:
            return {self.name: [None] * batch_size}

        sub_queries = [queries[idx] for idx in indices]
        sub_prompts = [prompts[idx] for idx in indices]
        sub_labels = [labels[idx] for idx in indices]
        values = self.component(sub_queries, sub_prompts, sub_labels)

        scattered: Dict[str, List[Optional[float]]] = {}
        for key, arr in values.items():
            if len(arr) != len(indices):
                raise ValueError(f"{key} len {len(arr)} != {len(indices)}")
            scattered[key] = [None] * batch_size
            for offset, idx in enumerate(indices):
                scattered[key][idx] = float(arr[offset])
        return scattered


class HeuristicWorkerGroup:
    def __init__(self, name: str, worker_actors: Sequence[ray.actor.ActorHandle]):
        if not worker_actors:
            raise ValueError(f"Heuristic worker group {name} requires at least one actor")
        self.name = name
        self.worker_actors = list(worker_actors)
        self._cursor = 0

    def dispatch(
        self, queries: List[str], prompts: List[str], labels: List[str]
    ) -> List[Tuple[int, int, ray.ObjectRef]]:
        batch_size = len(prompts)
        if batch_size == 0:
            return []

        num_chunks = min(len(self.worker_actors), batch_size)
        chunk_size = (batch_size + num_chunks - 1) // num_chunks
        actor_offset = self._cursor % len(self.worker_actors)
        self._cursor += num_chunks

        refs: List[Tuple[int, int, ray.ObjectRef]] = []
        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, batch_size)
            if start >= end:
                break
            actor = self.worker_actors[(actor_offset + chunk_idx) % len(self.worker_actors)]
            refs.append(
                (
                    start,
                    end,
                    actor.score_batch.remote(
                        queries[start:end],
                        prompts[start:end],
                        labels[start:end],
                    ),
                )
            )
        return refs

    def collect(
        self,
        refs: List[Tuple[int, int, ray.ObjectRef]],
        batch_size: int,
        results: Optional[List[Any]] = None,
    ) -> Dict[str, List[Optional[float]]]:
        if not refs:
            return {self.name: [None] * batch_size}

        if results is None:
            results = ray.get([ref for _, _, ref in refs])
        merged: Dict[str, List[Optional[float]]] = {}
        for (start, end, _), partial in zip(refs, results):
            chunk_size = end - start
            for key, arr in partial.items():
                if len(arr) != chunk_size:
                    raise ValueError(f"{key} len {len(arr)} != {chunk_size}")
                merged.setdefault(key, [None] * batch_size)
                merged[key][start:end] = arr

        merged.setdefault(self.name, [None] * batch_size)
        return merged


@ray.remote
class RewardHeadActor:
    """Aggregate heuristic-worker outputs into final ES rewards."""

    def __init__(self, heuristic_groups: Sequence[HeuristicWorkerGroup], expected_log_keys: Sequence[str]):
        self.heuristic_groups = list(heuristic_groups)
        self.expected_log_keys = list(expected_log_keys)

    def score(
        self, queries: List[str], prompts: List[str], labels: List[str]
    ) -> Dict[str, List[float] | Dict[str, List[float]]]:
        batch_size = len(prompts)
        total_rewards = [0.0] * batch_size
        extra_logs: Dict[str, List[Optional[float]]] = {}

        groups_and_refs: List[Tuple[HeuristicWorkerGroup, List[Tuple[int, int, ray.ObjectRef]]]] = []
        for group in self.heuristic_groups:
            refs = group.dispatch(queries, prompts, labels)
            groups_and_refs.append((group, refs))

        flat_refs: List[ray.ObjectRef] = []
        for _, refs in groups_and_refs:
            flat_refs.extend(ref for _, _, ref in refs)

        flat_results = ray.get(flat_refs) if flat_refs else []

        offset = 0
        for group, refs in groups_and_refs:
            n = len(refs)
            chunk = flat_results[offset : offset + n]
            offset += n
            values = group.collect(refs, batch_size, results=chunk)
            merge_optional_values(total_rewards, extra_logs, values)

        sanitized_extra_logs = sanitize_extra_logs(extra_logs, batch_size, self.expected_log_keys)
        return {
            "rewards": total_rewards,
            "scores": list(total_rewards),
            "extra_logs": sanitized_extra_logs,
        }


class RewardGraphManager:
    """Trainer-facing round-robin dispatcher over reward head actors."""

    def __init__(self, head_actors: Sequence[ray.actor.ActorHandle], expected_log_keys: Sequence[str]):
        if not head_actors:
            raise ValueError("Reward graph requires at least one reward head actor")
        self.head_actors = list(head_actors)
        self.expected_log_keys = list(expected_log_keys)
        self._cursor = 0

    def async_score(self, queries: List[str], prompts: List[str], labels: List[str]) -> ray.ObjectRef:
        actor = self.head_actors[self._cursor % len(self.head_actors)]
        self._cursor += 1
        return actor.score.remote(queries, prompts, labels)


def build_reward_graph(
    heuristics_path: str,
    *,
    heuristic_replicas: int = 1,
    reward_num_nodes: int = 1,
    reward_num_cpus_per_node: float = 1.0,
    reward_num_gpus_per_node: float = 0.0,
) -> RewardGraphManager:
    heuristic_classes = load_heuristic_classes(heuristics_path)
    expected_log_keys = [cls.__name__ for cls in heuristic_classes]

    heuristic_groups = []
    for heuristic_cls in heuristic_classes:
        worker_actors = []
        worker_cpus = float(getattr(heuristic_cls, "cpus", 1))
        worker_gpus = float(getattr(heuristic_cls, "gpus", 0))
        for _ in range(max(1, heuristic_replicas)):
            worker_actors.append(
                HeuristicWorkerActor.options(num_cpus=worker_cpus, num_gpus=worker_gpus).remote(
                    heuristics_path, heuristic_cls.__name__
                )
            )
        heuristic_groups.append(HeuristicWorkerGroup(heuristic_cls.__name__, worker_actors))

    head_actors = []
    for _ in range(max(1, reward_num_nodes)):
        head_actors.append(
            RewardHeadActor.options(
                num_cpus=reward_num_cpus_per_node,
                num_gpus=reward_num_gpus_per_node,
            ).remote(heuristic_groups, expected_log_keys)
        )

    return RewardGraphManager(head_actors, expected_log_keys)
