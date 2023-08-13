import os

from datasets import Dataset, interleave_datasets, load_dataset
from transformers import AutoTokenizer

from openllama2.utils import DeepspeedStrategy

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def get_tokenizer(pretrain, model, padding_side='left', strategy=None, use_fast=True):
    tokenizer = AutoTokenizer.from_pretrained(pretrain, trust_remote_code=True)
    tokenizer.padding_side = padding_side

    special_tokens_dict = dict()
    if tokenizer.pad_token is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
        strategy.print('add pad_token')
        tokenizer.add_special_tokens(special_tokens_dict)

    assert tokenizer.pad_token_id != tokenizer.eos_token_id
    return tokenizer


def get_strategy(args):
    if 'seed' not in args:
        args.seed = 42
    if 'max_norm' not in args:
        args.max_norm = 1.0
    if 'accumulated_gradient' not in args:
        args.accumulated_gradient = 1
    if 'train_batch_size' not in args:
        args.train_batch_size = 1
    if 'local_rank' not in args:
        args.local_rank = -1
    if 'bf16' not in args:
        args.bf16 = True
    if 'inference_tp_size' not in args:
        args.inference_tp_size = 1
    if 'adam_offload' not in args:
        args.adam_offload = False
    if 'zpg' not in args:
        args.zpg = 8

    # max_out_tokens for DS inference
    if 'max_len' in args and args.max_len is not None:
        args.max_out_tokens = args.max_len
    elif 'generate_max_len' in args and 'prompt_max_len' in args:
        args.max_out_tokens = args.prompt_max_len + args.generate_max_len
    else:
        raise Exception("Deepspeed config: Invalid max_out_tokens")

    strategy = DeepspeedStrategy(seed=args.seed, 
                                    max_norm=args.max_norm,
                                    accumulated_gradient=args.accumulated_gradient, 
                                    train_batch_size=args.train_batch_size,
                                    zero_stage=args.zero_stage,
                                    max_out_tokens=args.max_out_tokens,
                                    inference_tp_size=args.inference_tp_size,
                                    args=args)

    return strategy


def blending_datasets(datasets, probabilities, strategy=None, seed=42, max_count=2000000, return_eval=True):
    datasets = datasets.split(',')
    probabilities = list(map(float, probabilities.split(',')))
    assert len(probabilities) == len(datasets)

    train_data_list = []
    eval_data_list = []
    for i, dataset in enumerate(datasets):
        data = load_dataset(dataset.strip())
        if "train" in data:
            train_data_list.append(data["train"].select(range(min(max_count, len(data["train"])))))
        else:
            train_data_list.append(data.select(range(min(max_count, len(data)))))

        if return_eval:
            if 'test' in data:
                eval_data = data["test"].select(range(min(int(max_count * 0.1), len(data["test"]))))
            elif 'validation' in data:
                eval_data = data["validation"].select(range(min(int(max_count * 0.1), len(data["validation"]))))
            else:
                eval_data = Dataset.from_dict({})
            eval_data_list.append(eval_data)
    
    # merge datasets
    train_dataset = interleave_datasets(train_data_list, probabilities=probabilities, seed=seed)
    if return_eval:
        eval_dataset = interleave_datasets(eval_data_list, probabilities=probabilities, seed=seed)
        return train_dataset, eval_dataset
    else:
        return train_dataset


