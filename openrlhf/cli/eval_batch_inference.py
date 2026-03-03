"""
A script for evaluating the performance of run, defined as a directory containing the model's checkpoints.

"""

import argparse
import gc
import json
import os
import re
import subprocess as sp
from pathlib import Path

import pandas as pd
import torch
import wandb
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def clear_cuda_memory():
    """Clear CUDA memory cache and run garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def merge_lora_to_base(ckpt_path: Path) -> Path:
    """
    Merge LoRA weights with the base model and save to SLURM_TMPDIR.
    The base model path is retrieved from the adapter's config.

    Args:
        ckpt_path: Path to the LoRA checkpoint directory

    Returns:
        Path to the merged model saved in SLURM_TMPDIR
    """
    tmpdir = os.environ.get("SLURM_TMPDIR", "/tmp")
    merged_path = Path(tmpdir) / f"{ckpt_path.name}_merged"

    if merged_path.exists():
        print(f"Merged model already exists at {merged_path}, skipping merge.")
        return merged_path

    # Load the adapter config to get the base model path
    adapter_config_path = ckpt_path / "adapter_config.json"
    if not adapter_config_path.exists():
        raise FileNotFoundError(f"adapter_config.json not found in {ckpt_path}")

    with open(adapter_config_path) as f:
        adapter_config = json.load(f)

    base_model_path = adapter_config.get("base_model_name_or_path")
    if not base_model_path:
        raise ValueError(f"base_model_name_or_path not found in adapter_config.json at {ckpt_path}")

    print(f"Loading base model from {base_model_path}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    print(f"Loading LoRA weights from {ckpt_path}...")
    model = PeftModel.from_pretrained(base_model, str(ckpt_path))

    print("Merging LoRA weights with base model...")
    merged_model = model.merge_and_unload()

    print(f"Saving merged model to {merged_path}...")
    merged_model.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)

    # Free memory
    del base_model, model, merged_model
    print(f"Merged model saved to {merged_path}")
    return merged_path


def process_scored_generations(scored_path: Path) -> pd.DataFrame:
    # Open the jsonl file and read the scored generations
    step = re.search(r"global_step(\d+)_hf", scored_path.stem)
    if step is not None:
        step = int(step.group(1))
    else:
        step = 0
    dataframe = []

    with scored_path.open() as f:
        for line in f:
            item = json.loads(line)
            found_key = None
            for key in ["generation_verifier_metadata", "mol_prop_verifier_metadata", "reaction_verifier_metadata"]:
                if item["reward_meta"].get(key) is not None:
                    found_key = key
                    break
            if found_key is None:
                print("Warning: No metadata key found in reward_meta for item, skipping.")
                continue
            dataframe.append(item["reward_meta"][found_key])
            dataframe[-1]["reward"] = item["reward"]
            dataframe[-1]["prompt_id"] = item["metadata"].get("prompt_id", None)

            completion = item["output"]
            dataframe[-1]["n_tokens"] = (
                len(completion) // 4
            )  # Approximate token count (assuming 4 characters per token on average)

    # Convert the list of scored generations to a pandas DataFrame
    df = pd.DataFrame(dataframe)
    df["path"] = str(scored_path)
    df["global_step"] = step
    wandb_log_from_df(df, step)
    return df


def wandb_log_from_df(df: pd.DataFrame, step: int):
    # Log mean-reward for the scored generations
    mean_reward = df.groupby("prompt_id")["reward"].mean().mean()
    mean_n_tokens = df.groupby("prompt_id")["n_tokens"].mean().mean()
    average_validity = df["smiles_extraction_failure"].apply(lambda x: int(x == "")).mean()

    def get_uniqueness(group: pd.DataFrame) -> float:
        smis = group[group["smiles_extraction_failure"] == ""]["all_smi"].apply(lambda x: x[-1])
        return smis.nunique() / len(smis) if len(smis) > 0 else 0.0

    average_uniqueness = df.groupby("prompt_id").apply(get_uniqueness).mean()

    wandb.log(
        {
            "mean_reward": mean_reward,
            "mean_n_tokens": mean_n_tokens,
            "average_validity": average_validity,
            "average_uniqueness": average_uniqueness,
        },
        step=step,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument(
        "--ckpt_path", "-c", type=str, required=True, help="Path to the directory containing model checkpoints"
    )
    parser.add_argument("--dashboard_port", type=int, default=8265, help="Port for Ray dashboard")
    args = parser.parse_args()

    path = Path(args.ckpt_path)
    wandb.init(project="openrlhf-eval", name=f"eval_{path.parent.stem}")
    dfs = []
    # Sort checkpoints by global_step number for consistent wandb logging
    checkpoints = sorted(
        path.glob("*_hf"),
        key=lambda p: (
            int(re.search(r"global_step(\d+)_hf", p.name).group(1)) if re.search(r"global_step(\d+)_hf", p.name) else 0
        ),
    )
    for ckpt in checkpoints:
        print(f"Evaluating checkpoint: {ckpt}")

        # Merge LoRA weights with base model and save to SLURM_TMPDIR
        merged_ckpt = merge_lora_to_base(ckpt)

        # Free CUDA memory after merging
        clear_cuda_memory()

        out_path = str(ckpt.parent / f"{ckpt.stem}_eval_results")
        cmd = [
            "ray",
            "job",
            "submit",
            f"--address=http://127.0.0.1:{args.dashboard_port}",
            '--runtime-env-json={"setup_commands": ["wandb offline"]}',
            "--",
            "python3",
            "-m",
            "openrlhf.cli.batch_inference",
            "--config",
            args.config,
            "--pretrain",
            str(merged_ckpt),
            "--output_path",
            out_path,
        ]
        print(f"- Running batch inference with command: {' '.join(cmd)}")
        sp.run(cmd)

        # Free CUDA memory after batch inference
        clear_cuda_memory()

        print(f"Scoring results for checkpoint: {ckpt}")
        cmd = [
            "python",
            "-m",
            "mol_gen_docking.score_completions",
            "--input_file",
            out_path + ".jsonl",
            "--mol-generation",
        ]
        print(f"- Scoring generations with command: {' '.join(cmd)}")
        sp.run(cmd)

        # Free CUDA memory after scoring
        clear_cuda_memory()

        scored_path = Path(out_path + "_scored.jsonl")
        if scored_path.exists():
            print(f"Processing scored generations for checkpoint: {ckpt}")
            df = process_scored_generations(scored_path)
            dfs.append(df)
        else:
            print(f"Scored generations file not found for checkpoint: {ckpt}, expected at: {scored_path}")
    if dfs:
        all_results_df = pd.concat(dfs, ignore_index=True)
        all_results_df.to_csv(path / "all_scored_generations.csv", index=False)
        print(f"Saved all scored generations to: {path / 'all_scored_generations.csv'}")
        # Log to wandb as a Table
        wandb.log({"scored_generations": wandb.Table(dataframe=all_results_df)})
