import os
from pathlib import Path

from datasets import Dataset, interleave_datasets, load_dataset
from transformers import AutoTokenizer

from openrlhf.utils import DeepspeedStrategy

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def get_tokenizer(pretrain, model, padding_side="left", strategy=None, use_fast=True):
    tokenizer = AutoTokenizer.from_pretrained(pretrain, trust_remote_code=True, use_fast=use_fast)
    tokenizer.padding_side = padding_side
    # NOTE: When enable vLLM, do not resize_token_embeddings, or the vocab size will mismatch with vLLM.
    # https://github.com/facebookresearch/llama-recipes/pull/196
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer


def get_strategy(args):
    strategy = DeepspeedStrategy(
        seed=getattr(args, "seed", 42),
        max_norm=getattr(args, "max_norm", 1.0),
        micro_train_batch_size=getattr(args, "micro_train_batch_size", 1),
        train_batch_size=getattr(args, "train_batch_size", 128),
        zero_stage=args.zero_stage,
        bf16=getattr(args, "bf16", True),
        args=args,
    )
    return strategy


def blending_datasets(
    datasets,
    probabilities,
    strategy=None,
    seed=42,
    max_count=5000000,
    return_eval=True,
    stopping_strategy="first_exhausted",
    train_split="train",
    eval_split="test",
):
    datasets = datasets.split(",")
    probabilities = list(map(float, probabilities.split(",")))
    assert len(probabilities) == len(datasets)

    train_data_list = []
    eval_data_list = []
    for i, dataset in enumerate(datasets):
        dataset = dataset.strip()
        strategy.print(f"dataset: {dataset}")

        data_dir = dataset.split("@")[1].strip() if "@" in dataset else None
        dataset = dataset.split("@")[0].strip()
        dataset_basename = os.path.basename(dataset)

        ext = os.path.splitext(dataset)[-1]
        # local python script
        if ext == ".py" or (
            os.path.isdir(dataset) and os.path.exists(os.path.join(dataset, f"{dataset_basename}.py"))
        ):
            data = load_dataset(dataset, trust_remote_code=True)
            strategy.print(f"loaded {dataset} with python script")
        # local text file
        elif ext in [".json", ".jsonl", ".csv"]:
            ext = ext.lower().strip(".")
            if ext == "jsonl":
                ext = "json"
            data = load_dataset(ext, data_files=dataset)
        # remote/local folder or common file
        else:
            data = load_dataset(dataset, data_dir=data_dir)
            strategy.print(f"loaded {dataset} from files")

        if train_split and train_split in data:
            train_data = data[train_split].select(range(min(max_count, len(data[train_split]))))
        else:
            train_data = data.select(range(min(max_count, len(data))))
        train_data_list.append(train_data)

        if return_eval:
            if eval_split and eval_split in data:
                eval_data = data[eval_split].select(range(min(max_count, len(data[eval_split]))))
            # train will contains eval? TODO
            else:
                eval_data = train_data.select(range(min(max_count, int(len(train_data) * 0.03))))
            eval_data_list.append(eval_data)

    # merge datasets
    if strategy.is_rank_0():
        print(train_data_list)

    train_dataset = interleave_datasets(
        train_data_list,
        probabilities=probabilities,
        seed=seed,
        stopping_strategy=stopping_strategy,
    )
    if return_eval:
        eval_dataset = interleave_datasets(
            eval_data_list,
            probabilities=probabilities,
            seed=seed,
            stopping_strategy=stopping_strategy,
        )
        return train_dataset, eval_dataset
    else:
        return train_dataset
