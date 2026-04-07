import json
import os
import re
import shutil
from typing import Dict, Optional, Tuple

import ray

from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)

_RE_LATEST_HF_DIR = re.compile(r"^global_step[0-9]+_hf$")
_RE_BEST_HF_DIR = re.compile(r"^best_global_step[0-9]+_hf$")


def remove_previous_latest_hf_ckpts(ckpt_path: str) -> None:
    """Keep a single rolling latest checkpoint: drop any prior global_step{N}_hf dirs."""
    if not os.path.isdir(ckpt_path):
        return
    for name in os.listdir(ckpt_path):
        if not _RE_LATEST_HF_DIR.match(name):
            continue
        path = os.path.join(ckpt_path, name)
        if os.path.isdir(path):
            logger.info("Removing previous latest checkpoint: %s", path)
            shutil.rmtree(path, ignore_errors=True)


def remove_previous_best_hf_ckpts(ckpt_path: str) -> None:
    """Keep a single best checkpoint: drop any prior best_global_step{N}_hf dirs."""
    if not os.path.isdir(ckpt_path):
        return
    for name in os.listdir(ckpt_path):
        if not _RE_BEST_HF_DIR.match(name):
            continue
        path = os.path.join(ckpt_path, name)
        if os.path.isdir(path):
            logger.info("Removing previous best checkpoint: %s", path)
            shutil.rmtree(path, ignore_errors=True)


def write_trainer_state(
    ckpt_path: str,
    global_step: int,
    episode_idx: int,
    best_eval_metric_key: Optional[str],
    best_eval_metric_value: float,
    client_states: Dict,
) -> None:
    os.makedirs(ckpt_path, exist_ok=True)
    path = os.path.join(ckpt_path, "trainer_state.json")
    state = {
        "global_step": global_step,
        "epoch": episode_idx,
        "episode": episode_idx,
        "best_eval_metric_key": best_eval_metric_key,
        "best_eval_metric_value": best_eval_metric_value,
    }
    state.update(client_states)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)


def save_es_hf_checkpoint(
    *,
    args,
    tag: str,
    global_step: int,
    client_states: Dict,
    vllm_engines,
    pretrain: str,
    episode_idx: int,
    best_eval_metric_key: Optional[str],
    best_eval_metric_value: float,
) -> None:
    if args.vllm_tensor_parallel_size != 1:
        logger.warning(
            "ES HF checkpoint save requires vllm_tensor_parallel_size=1; skipping save for tag %s",
            tag,
        )
        return
    if tag.startswith("best_global_step"):
        remove_previous_best_hf_ckpts(args.ckpt_path)
    elif tag.startswith("global_step"):
        remove_previous_latest_hf_ckpts(args.ckpt_path)

    out_dir = os.path.join(args.ckpt_path, f"{tag}_hf")
    logger.info("Saving ES HF checkpoint to %s", out_dir)
    ray.get(vllm_engines[0].save_hf_checkpoint.remote(out_dir, pretrain))
    write_trainer_state(
        args.ckpt_path,
        global_step,
        episode_idx,
        best_eval_metric_key,
        best_eval_metric_value,
        client_states,
    )


def detect_eval_metric_key(
    best_eval_metric_key: Optional[str],
    eval_metrics: Dict,
) -> Tuple[Optional[str], Optional[str]]:
    if best_eval_metric_key == "none":
        return None, best_eval_metric_key
    if best_eval_metric_key:
        return (
            best_eval_metric_key if best_eval_metric_key in eval_metrics else None,
            best_eval_metric_key,
        )
    for key in sorted(eval_metrics):
        if key.endswith("_pass1"):
            return key, key
    return None, best_eval_metric_key


def init_checkpoint_states(load_checkpoint: bool, ckpt_path: str) -> Tuple[Dict, Optional[str], Optional[float]]:
    state = {
        "episode": 0,
        "epoch": 0,
        "global_step": 0,
        "total_consumed_prompts": 0,
        "data_loader_state_dict": {},
    }
    if not load_checkpoint:
        return state, None, None

    path = os.path.join(ckpt_path, "trainer_state.json")
    if not os.path.isfile(path):
        return state, None, None

    try:
        with open(path, encoding="utf-8") as handle:
            loaded = json.load(handle)
        state["global_step"] = int(loaded.get("global_step", 0))
        epoch = int(loaded.get("epoch", loaded.get("episode", 0)))
        state["epoch"] = epoch
        state["episode"] = epoch
        state["total_consumed_prompts"] = loaded.get("total_consumed_prompts", 0)
        state["data_loader_state_dict"] = loaded.get("data_loader_state_dict", {})
        best_key = loaded.get("best_eval_metric_key")
        best_value = float(loaded["best_eval_metric_value"]) if "best_eval_metric_value" in loaded else None
        return state, best_key, best_value
    except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
        logger.warning("Could not load trainer_state.json: %s", exc)
        return state, None, None
