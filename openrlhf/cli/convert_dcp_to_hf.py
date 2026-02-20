"""Convert a DCP (Distributed Checkpoint) critic model to HuggingFace safetensors format.

Usage:
    torchrun --nproc_per_node=8 -m openrlhf.cli.convert_dcp_to_hf \
        --pretrain Qwen/Qwen2.5-Math-7B \
        --ckpt_path ./ckpt/checkpoints/_critic \
        --output_dir ./output/critic_hf \
        --value_head_prefix score
"""

import argparse

from openrlhf.models import get_llm_for_sequence_regression
from openrlhf.utils import convert_to_torch_dtype, get_strategy, get_tokenizer


def convert(args):
    strategy = get_strategy(args)
    strategy.setup_distributed()

    # 1. Build critic model structure on meta device
    model = get_llm_for_sequence_regression(
        args.pretrain,
        "critic",
        normalize_reward=args.normalize_reward,
        attn_implementation=args.attn_implementation,
        torch_dtype=convert_to_torch_dtype("fp32"),
        init_device="meta_structure",
        value_head_prefix=args.value_head_prefix,
        packing_samples=args.packing_samples,
    )

    # 2. FSDP2 prepare (TP + FSDP sharding) + load base pretrained weights
    model = strategy.prepare(model)
    strategy.load_pretrained(
        model,
        args.pretrain,
        init_value_head=True,
        value_head_prefix=args.value_head_prefix,
    )

    # 3. Load DCP checkpoint (model weights only, skip optimizer/scheduler)
    tag, client_state = strategy.load_dcp_model(
        model,
        args.ckpt_path,
        tag=args.tag,
        load_module_only=True,
    )
    strategy.print(f"Loaded DCP checkpoint: tag={tag}")

    # 4. Save value_head_prefix in config so from_pretrained can find it
    unwrapped = strategy._unwrap_model(model)
    unwrapped.config.value_head_prefix = args.value_head_prefix

    # 5. Save as HF safetensors
    tokenizer = get_tokenizer(args.pretrain, unwrapped, "left", strategy)
    strategy.save_hf_model(model, tokenizer, args.output_dir)
    strategy.print(f"Saved HF checkpoint to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DCP critic checkpoint to HuggingFace format")

    # Required
    parser.add_argument("--pretrain", type=str, required=True, help="Base model path (for model structure + tokenizer/config)")
    parser.add_argument("--ckpt_path", type=str, required=True, help="DCP checkpoint directory (e.g. ./ckpt/checkpoints/_critic)")
    parser.add_argument("--output_dir", type=str, required=True, help="HF output directory")

    # DCP options
    parser.add_argument("--tag", type=str, default=None, help="DCP checkpoint tag (default: read from 'latest')")

    # Model options
    parser.add_argument("--value_head_prefix", type=str, default="score", help="Value head name")
    parser.add_argument("--normalize_reward", action="store_true", default=False, help="Enable reward normalization")
    parser.add_argument("--packing_samples", action="store_true", default=False, help="Enable packing samples")

    # FSDP2 / distributed
    parser.add_argument("--param_dtype", type=str, default="bf16", choices=["bf16", "fp16"], help="Model data type")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2", help="Attention implementation")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank (for torchrun)")
    parser.add_argument("--fsdp2_tp_size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--fsdp2_cp_size", type=int, default=1, help="Context parallel size")
    parser.add_argument("--fsdp2_cpu_offload", action="store_true", default=False, help="Enable CPU offload")
    parser.add_argument("--fsdp2_reshard_after_forward", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sequence_parallel", action="store_true", default=False)
    parser.add_argument("--tp_shard_logits", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)

    # HF export options
    parser.add_argument("--hf_max_shard_size_gb", type=float, default=5, help="Max shard size (GB) for HF safetensors")

    args = parser.parse_args()

    # Set defaults expected by get_strategy but not needed for conversion
    args.full_determinism = False
    args.max_norm = 1.0
    args.micro_train_batch_size = 1
    args.train_batch_size = 1
    args.lora_rank = 0
    args.use_dynamic_batch = False

    convert(args)
