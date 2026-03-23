import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch


def _to_python_scalar(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    return value


def _safe_mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _safe_max(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return max(values)


def build_eval_records(samples_list: List, prompt_metadata: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    prompt_offsets = defaultdict(int)

    for samples in samples_list:
        batch_size = len(samples.prompts)
        for idx in range(batch_size):
            prompt = samples.prompts[idx]
            prompt_index = prompt_offsets[prompt]
            prompt_offsets[prompt] += 1

            metadata_list = prompt_metadata.get(prompt, [])
            metadata = metadata_list[prompt_index] if prompt_index < len(metadata_list) else {}
            info = samples.info or {}

            reward_value = None
            if samples.rewards is not None:
                reward_value = _to_python_scalar(samples.rewards[idx])
            if reward_value is None and "reward" in info:
                reward_tensor = info["reward"]
                reward_value = _to_python_scalar(reward_tensor[idx])

            response_length = _to_python_scalar(info["response_length"][idx]) if "response_length" in info else None
            total_length = _to_python_scalar(info["total_length"][idx]) if "total_length" in info else None
            truncated = _to_python_scalar(info["truncated"][idx]) if "truncated" in info else None
            score = _to_python_scalar(info["score"][idx]) if "score" in info else None

            record = {
                "prompt": prompt,
                "label": samples.labels[idx] if idx < len(samples.labels) else None,
                "reward": reward_value,
                "response_length": response_length,
                "total_length": total_length,
                "truncated": bool(truncated) if truncated is not None else None,
                "score": score,
                "datasource": metadata.get("datasource", "unknown"),
                "eval_index": metadata.get("eval_index"),
            }

            for key, value in info.items():
                if key in {"reward", "response_length", "total_length", "truncated", "score"}:
                    continue
                if isinstance(value, torch.Tensor) and value.ndim > 0 and idx < len(value):
                    record[key] = _to_python_scalar(value[idx])

            records.append(record)

    return records


def aggregate_eval_metrics(records: List[Dict[str, Any]], n_samples_per_prompt: int) -> Dict[str, Optional[float]]:
    logs: Dict[str, Optional[float]] = {}

    rewards = [float(r["reward"]) for r in records if r.get("reward") is not None]
    response_lengths = [float(r["response_length"]) for r in records if r.get("response_length") is not None]
    truncated_flags = [1.0 if r.get("truncated") else 0.0 for r in records if r.get("truncated") is not None]

    logs["reward_mean"] = _safe_mean(rewards)
    logs["reward_max"] = _safe_max(rewards)
    logs["response_length_mean"] = _safe_mean(response_lengths)
    logs["response_length_max"] = _safe_max(response_lengths)
    logs["truncated_rate"] = _safe_mean(truncated_flags)
    logs["num_eval_samples"] = float(len(records)) if records else 0.0

    grouped_by_prompt = defaultdict(list)
    grouped_by_datasource = defaultdict(list)
    for record in records:
        prompt_key = (record.get("datasource", "unknown"), record.get("prompt"))
        grouped_by_prompt[prompt_key].append(record)
        grouped_by_datasource[record.get("datasource", "unknown")].append(record)

    pass_at_k_values = []
    pass1_values = []
    for prompt_records in grouped_by_prompt.values():
        valid_prompt_records = [r for r in prompt_records if r.get("reward") is not None]
        for start in range(0, len(valid_prompt_records), n_samples_per_prompt):
            prompt_group = valid_prompt_records[start : start + n_samples_per_prompt]
            prompt_rewards = [float(r["reward"]) for r in prompt_group]
            if not prompt_rewards:
                continue
            pass1_values.append(sum(prompt_rewards) / len(prompt_rewards))
            if n_samples_per_prompt > 1:
                pass_at_k_values.append(max(prompt_rewards))

    logs["pass1"] = _safe_mean(pass1_values)
    logs[f"pass{n_samples_per_prompt}"] = _safe_mean(pass_at_k_values) if n_samples_per_prompt > 1 else logs["pass1"]

    for datasource, datasource_records in grouped_by_datasource.items():
        ds_rewards = [float(r["reward"]) for r in datasource_records if r.get("reward") is not None]
        ds_lengths = [float(r["response_length"]) for r in datasource_records if r.get("response_length") is not None]
        ds_truncated = [
            1.0 if r.get("truncated") else 0.0 for r in datasource_records if r.get("truncated") is not None
        ]

        logs[f"{datasource}_reward_mean"] = _safe_mean(ds_rewards)
        logs[f"{datasource}_response_length_mean"] = _safe_mean(ds_lengths)
        logs[f"{datasource}_truncated_rate"] = _safe_mean(ds_truncated)

        datasource_prompt_groups = defaultdict(list)
        for record in datasource_records:
            datasource_prompt_groups[record.get("prompt")].append(record)

        ds_pass1 = []
        ds_passk = []
        for prompt_records in datasource_prompt_groups.values():
            valid_prompt_records = [r for r in prompt_records if r.get("reward") is not None]
            for start in range(0, len(valid_prompt_records), n_samples_per_prompt):
                prompt_group = valid_prompt_records[start : start + n_samples_per_prompt]
                prompt_rewards = [float(r["reward"]) for r in prompt_group]
                if not prompt_rewards:
                    continue
                ds_pass1.append(sum(prompt_rewards) / len(prompt_rewards))
                if n_samples_per_prompt > 1:
                    ds_passk.append(max(prompt_rewards))

        logs[f"{datasource}_pass1"] = _safe_mean(ds_pass1)
        logs[f"{datasource}_pass{n_samples_per_prompt}"] = (
            _safe_mean(ds_passk) if n_samples_per_prompt > 1 else logs[f"{datasource}_pass1"]
        )

    return logs


def get_eval_sample_preview(records: List[Dict[str, Any]], max_samples: int = 8) -> List[Dict[str, Any]]:
    preview = []
    for record in records[:max_samples]:
        preview.append(
            {
                "datasource": record.get("datasource"),
                "prompt": record.get("prompt"),
                "label": record.get("label"),
                "reward": record.get("reward"),
                "response_length": record.get("response_length"),
                "truncated": record.get("truncated"),
            }
        )
    return preview


def save_eval_samples(
    records: List[Dict[str, Any]], save_dir: str, global_step: int, max_samples: int = 128
) -> Optional[str]:
    if not records or not save_dir:
        return None

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"eval_samples_step_{global_step}.jsonl")
    with open(save_path, "w", encoding="utf-8") as f:
        for record in records[:max_samples]:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return save_path
