import torch
from transformers import AutoTokenizer


def convert_to_torch_dtype(param_dtype: str) -> torch.dtype:
    """Convert param_dtype string to torch.dtype.

    Args:
        param_dtype: One of "bf16", "fp16"

    Returns:
        Corresponding torch.dtype (bfloat16, float16)
    """
    if param_dtype == "bf16":
        return torch.bfloat16
    elif param_dtype == "fp16":
        return torch.float16
    else:
        raise ValueError(f"Invalid param_dtype: {param_dtype}")


def get_strategy(args):
    from openrlhf.utils.deepspeed import DeepspeedStrategy

    strategy = DeepspeedStrategy(
        seed=getattr(args.train, "seed", 42),
        full_determinism=getattr(args.train, "full_determinism_enable", False),
        max_norm=getattr(args, "max_norm", 1.0),
        micro_train_batch_size=getattr(args.train, "micro_batch_size", 1),
        train_batch_size=getattr(args.train, "batch_size", 128),
        zero_stage=args.ds.zero_stage,
        args=args,
    )
    return strategy


def get_tokenizer(pretrain, model, padding_side="left", strategy=None, use_fast=True):
    tokenizer = AutoTokenizer.from_pretrained(pretrain, trust_remote_code=True, use_fast=use_fast)
    tokenizer.padding_side = padding_side
    # NOTE: When enable vLLM, do not resize_token_embeddings, or the vocab size will mismatch with vLLM.
    # https://github.com/facebookresearch/llama-recipes/pull/196
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if model is not None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer


