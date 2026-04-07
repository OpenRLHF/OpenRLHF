from __future__ import annotations

import importlib.util
import inspect
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import torch

Values = Dict[str, List[float]]
OptionalValues = Dict[str, List[Optional[float]]]


class RewardComponent:
    """Duck-typed reward component interface."""

    def should_call(self, label: str) -> bool:
        return True

    def __call__(self, queries: List[str], prompts: List[str], labels: List[str]) -> Values:
        raise NotImplementedError


class NamedRewardComponent:
    def __init__(self, name: str, component: RewardComponent):
        self.name = name
        self.component = component

    def should_call(self, label: str) -> bool:
        return self.component.should_call(label)

    def __call__(self, queries: List[str], prompts: List[str], labels: List[str]) -> Values:
        values = self.component(queries, prompts, labels)
        if len(values) != 1:
            raise ValueError(f"{self.name} must return exactly one reward key, got: {list(values.keys())}")
        return {self.name: next(iter(values.values()))}


class Aggregator:
    """Score samples using component routing based on should_call."""

    def __init__(self, components: List[RewardComponent], device: torch.device):
        self.components = components
        self.device = device

    @torch.no_grad()
    def score(
        self,
        queries: List[str],
        prompts: List[str],
        labels: List[str],
    ) -> Tuple[List[float], OptionalValues]:
        n = len(prompts)
        extra_logs: OptionalValues = {}
        total_rewards = torch.zeros(n, device=self.device, dtype=torch.float32)

        for comp in self.components:
            indices = [i for i, label in enumerate(labels) if comp.should_call(label)]
            if not indices:
                continue

            sub_queries = [queries[i] for i in indices]
            sub_prompts = [prompts[i] for i in indices]
            sub_labels = [labels[i] for i in indices]
            vals = comp(sub_queries, sub_prompts, sub_labels)

            integrate_component_outputs(total_rewards, extra_logs, indices, vals)

        return total_rewards.detach().cpu().tolist(), extra_logs


def integrate_component_outputs(
    total_rewards: torch.Tensor,
    extra_logs: OptionalValues,
    indices: List[int],
    vals: Values,
) -> None:
    sub_n = len(indices)
    for key, arr in vals.items():
        if len(arr) != sub_n:
            raise ValueError(f"{key} len {len(arr)} != {sub_n}")

        v = torch.tensor(arr, device=total_rewards.device, dtype=torch.float32)
        idx_tensor = torch.tensor(indices, device=total_rewards.device, dtype=torch.long)
        total_rewards.scatter_add_(0, idx_tensor, v)

        if key not in extra_logs:
            extra_logs[key] = [None] * total_rewards.numel()

        for j, idx in enumerate(indices):
            extra_logs[key][idx] = arr[j]


def merge_optional_values(total_rewards: List[float], extra_logs: OptionalValues, values: OptionalValues) -> None:
    for key, arr in values.items():
        if key not in extra_logs:
            extra_logs[key] = [None] * len(total_rewards)
        if len(arr) != len(total_rewards):
            raise ValueError(f"{key} len {len(arr)} != {len(total_rewards)}")
        for idx, value in enumerate(arr):
            if value is None:
                continue
            total_rewards[idx] += float(value)
            extra_logs[key][idx] = float(value)


def sanitize_extra_logs(
    extra_logs: OptionalValues,
    batch_size: int,
    expected_log_keys: Optional[List[str]] = None,
) -> Dict[str, List[float]]:
    # Use NaN for inactive samples so downstream means divide by applicable count, not batch size.
    sanitized = {k: [math.nan if v is None else float(v) for v in vals] for k, vals in extra_logs.items()}
    for key in sorted(set(expected_log_keys or [])):
        sanitized.setdefault(key, [math.nan] * batch_size)
    return sanitized


def load_heuristic_classes(heuristics_path: str) -> List[Type[RewardComponent]]:
    path = Path(heuristics_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Heuristics file not found: {heuristics_path}")

    spec = importlib.util.spec_from_file_location(f"_heuristics_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load heuristics file: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    heuristic_classes = getattr(module, "HEURISTICS", None)
    if not isinstance(heuristic_classes, list) or not heuristic_classes:
        raise ValueError(f"{path} must define a non-empty HEURISTICS list")

    seen_names: set[str] = set()
    for cls in heuristic_classes:
        if not inspect.isclass(cls):
            raise ValueError(f"HEURISTICS entries must be classes, got: {cls!r}")
        if cls.__name__ in seen_names:
            raise ValueError(f"Duplicate heuristic class name: {cls.__name__}")
        seen_names.add(cls.__name__)

    return heuristic_classes


def instantiate_heuristic(
    heuristic_cls: Type[RewardComponent],
    *,
    device: Optional[torch.device] = None,
) -> RewardComponent:
    sig = inspect.signature(heuristic_cls.__init__)
    kwargs: Dict[str, Any] = {}
    for name, param in list(sig.parameters.items())[1:]:
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if name == "device" and device is not None:
            kwargs["device"] = device
            continue
        if param.default is inspect.Parameter.empty:
            raise ValueError(f"{heuristic_cls.__name__} requires unsupported init arg: {name}")

    return heuristic_cls(**kwargs)


def load_heuristics(heuristics_path: str, device: torch.device) -> Tuple[List[RewardComponent], List[str]]:
    components: List[RewardComponent] = []
    expected_log_keys: List[str] = []
    for heuristic_cls in load_heuristic_classes(heuristics_path):
        components.append(
            NamedRewardComponent(heuristic_cls.__name__, instantiate_heuristic(heuristic_cls, device=device))
        )
        expected_log_keys.append(heuristic_cls.__name__)
    return components, expected_log_keys
