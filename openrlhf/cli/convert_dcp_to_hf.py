"""Convert a DCP (Distributed Checkpoint) model to HuggingFace safetensors format.

Usage:
    # Convert a critic checkpoint from the new step-level layout:
    torchrun --nproc_per_node=8 -m openrlhf.cli.convert_dcp_to_hf \
        --model_type critic \
        --model_name_or_path Qwen/Qwen2.5-Math-7B \
        --ckpt_path ./ckpt/dcp_ckpt/global_step_100/dcp_checkpoint/_critic \
        --output_dir ./output/critic_hf \
        --value_head_prefix score

    # Convert an actor checkpoint:
    torchrun --nproc_per_node=8 -m openrlhf.cli.convert_dcp_to_hf \
        --model_type actor \
        --model_name_or_path Qwen/Qwen2.5-Math-7B \
        --ckpt_path ./ckpt/dcp_ckpt/global_step_100/dcp_checkpoint/_actor \
        --output_dir ./output/actor_hf
"""

import argparse

from openrlhf.models import Actor, get_llm_for_sequence_regression
from openrlhf.utils import convert_to_torch_dtype, get_strategy, get_tokenizer


def convert(args):
    strategy = get_strategy(args)
    strategy.setup_distributed()

    # 1. Build model structure on meta device
    if args.model_type == "critic":
        model = get_llm_for_sequence_regression(
            args.model_name_or_path,
            "critic",
            normalize_reward=args.normalize_reward,
            attn_implementation=args.attn_implementation,
            torch_dtype=convert_to_torch_dtype("fp32"),
            value_head_prefix=args.value_head_prefix,
            packing_samples=args.packing_samples,
        )
    else:
        model = Actor(
            args.model_name_or_path,
            attn_implementation=args.attn_implementation,
            torch_dtype=convert_to_torch_dtype("fp32"),
            packing_samples=args.packing_samples,
        )

    # 2. FSDP2 prepare (TP + FSDP sharding)
    model = strategy.apply_parallelism(model)

    # 3. Materialize parameters by loading base HF weights.
    if args.model_type == "critic":
        strategy.load_hf_checkpoint(
            model,
            args.model_name_or_path,
            init_value_head=True,
            value_head_prefix=args.value_head_prefix,
        )
    else:
        strategy.load_hf_checkpoint(model, args.model_name_or_path)

    # 4. Load DCP model weights (skip optimizer/scheduler).
    strategy.load_dcp_checkpoint(
        model,
        args.ckpt_path,
        load_module_only=True,
    )
    strategy.print(f"Loaded DCP checkpoint: {args.ckpt_path}")

    # 5. Save value_head_prefix in config so from_pretrained can find it (critic only)
    unwrapped = strategy._unwrap_model(model)
    if args.model_type == "critic":
        unwrapped.config.value_head_prefix = args.value_head_prefix

    # 6. Save as HF safetensors
    tokenizer = get_tokenizer(args.model_name_or_path, unwrapped, "left", strategy)
    strategy.save_hf_checkpoint(model, tokenizer, args.output_dir)
    strategy.print(f"Saved HF checkpoint to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DCP actor/critic checkpoint to HuggingFace format")

    # Required
    parser.add_argument("--model_type", type=str, required=True, choices=["actor", "critic"], help="Checkpoint model type.")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Base model path (for model structure + tokenizer/config)")
    parser.add_argument("--ckpt_path", type=str, required=True, help="DCP checkpoint directory (e.g. ./ckpt/dcp_ckpt/global_step_100/dcp_checkpoint/_critic)")
    parser.add_argument("--output_dir", type=str, required=True, help="HF output directory")

    # Model options
    parser.add_argument("--value_head_prefix", type=str, default="score", help="Value head name (critic only).")
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
    parser.add_argument("--fsdp2_tp_sequence_parallel", action="store_true", default=False)
    parser.add_argument(
        "--fsdp2_tp_loss_parallel",
        action="store_true",
        default=False,
        help="Enable vocab-sharded lm_head logits and TP loss-parallel path (requires --fsdp2_tp_size > 1).",
    )
    parser.add_argument("--seed", type=int, default=42)

    # HF export options
    parser.add_argument(
        "--hf_max_shard_size_gb",
        type=float,
        default=5,
        help="Max size (in GB) per .safetensors shard file when saving HuggingFace checkpoints, e.g. 5 means each file <= 5GB.",
    )

    args = parser.parse_args()

    # Set defaults expected by get_strategy but not needed for conversion
    args.full_determinism = False
    args.max_norm = 1.0
    args.micro_train_batch_size = 1
    args.train_batch_size = 1
    args.use_dynamic_batch = False

    convert(args)
