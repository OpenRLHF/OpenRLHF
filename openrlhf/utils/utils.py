from typing import List

import torch
import torch.nn.functional as F
from packaging import version
from transformers import AutoTokenizer


_PRECISION_TO_DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def convert_to_dtype(precision: str) -> torch.dtype:
    try:
        return _PRECISION_TO_DTYPE[precision]
    except KeyError as exc:
        raise ValueError(f"Invalid precision: {precision!r}. Expected one of {tuple(_PRECISION_TO_DTYPE)}.") from exc


def get_strategy(args):
    dist_backend = getattr(args, "dist_backend", "deepspeed")

    if dist_backend == "fsdp2":
        torch_version = version.parse(torch.__version__.split("+")[0])
        if torch_version < version.parse("2.4.0"):
            raise RuntimeError(
                f"FSDP2 backend requires torch>=2.4.0. Detected torch=={torch.__version__}. "
                "Please upgrade PyTorch or use --dist_backend deepspeed."
            )
        try:
            from openrlhf.utils.fsdp import FSDP2Strategy  # type: ignore
        except Exception as e:
            raise RuntimeError(
                f"FSDP2 backend requested but not available: {e}. Please use --dist_backend deepspeed or install a recent PyTorch with FSDP2."
            )

        strategy = FSDP2Strategy(
            seed=getattr(args, "seed", 42),
            full_determinism=getattr(args, "full_determinism", False),
            max_norm=getattr(args, "max_norm", 1.0),
            micro_train_batch_size=getattr(args, "micro_train_batch_size", 1),
            train_batch_size=getattr(args, "train_batch_size", 128),
            args=args,
        )
        return strategy

    # default: deepspeed
    from openrlhf.utils.deepspeed import DeepspeedStrategy

    strategy = DeepspeedStrategy(
        seed=getattr(args, "seed", 42),
        full_determinism=getattr(args, "full_determinism", False),
        max_norm=getattr(args, "max_norm", 1.0),
        micro_train_batch_size=getattr(args, "micro_train_batch_size", 1),
        train_batch_size=getattr(args, "train_batch_size", 128),
        zero_stage=getattr(args, "zero_stage", 2),
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
        if model is not None:
            model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer


def convert_token_to_id(token, tokenizer):
    if isinstance(token, str):
        token = tokenizer.encode(token, add_special_tokens=False)
        assert len(token) == 1
        return token[0]
    else:
        raise ValueError("token should be int or str")


def zero_pad_sequences(
    sequences: List[torch.Tensor], side: str = "left", value: int = 0, stack: bool = False
) -> torch.Tensor:
    assert side in ("left", "right")
    max_len = max(seq.size(-1) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(-1)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding, value=value))
    if stack:
        return torch.stack(padded_sequences, dim=0)
    else:
        return torch.cat(padded_sequences, dim=0)


def remove_pad_token(input_ids: torch.Tensor, attention_mask: torch.Tensor):
    """Remove the pad token. Return tensors and not lists.

    Args:
        input_ids shape: [bs, seq_length]
        attention_mask shape: [bs, seq_length]
    Returns:
        no_padding_batch(List[Tensor[int]]): contains the rmpad token ids per query.
    """
    no_padding_batch = []
    for ids, mask in zip(input_ids, attention_mask):
        # Fix for both left and right padding
        no_padding_batch.append((ids[mask.bool()]))
    return no_padding_batch
