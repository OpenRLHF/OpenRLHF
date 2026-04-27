import argparse
import math
import os
from datetime import datetime

from openrlhf.datasets import SFTDataset
from openrlhf.datasets.utils import blending_datasets
from openrlhf.models import Actor
from openrlhf.trainer.sft_trainer import SFTTrainer
from openrlhf.utils import get_strategy, get_tokenizer


def train(args):
    strategy = get_strategy(args)
    strategy.setup_distributed()

    model = Actor(
        args.model.model_name_or_path,
        attn_implementation=args.fsdp.attn_implementation,
        param_dtype=args.fsdp.param_dtype,
        load_in_4bit=args.fsdp.load_in_4bit,
        lora_rank=args.fsdp.lora.rank,
        lora_alpha=args.fsdp.lora.alpha,
        target_modules=args.fsdp.lora.target_modules,
        lora_dropout=args.fsdp.lora.dropout,
        device_mesh=strategy.device_mesh,
        moe_mesh=strategy.moe_mesh,
        distributed_config=strategy.distributed_config,
        moe_config=strategy.moe_config,
        activation_checkpointing=args.model.gradient_checkpointing_enable,
        packing_samples=args.fsdp.packing_samples,
        use_liger_kernel=args.fsdp.use_liger_kernel,
    )
    tokenizer = get_tokenizer(
        args.model.model_name_or_path, model.model, "right", strategy, use_fast=not args.data.disable_fast_tokenizer
    )
    strategy.print(model)

    train_data = blending_datasets(
        args.data.dataset,
        args.data.dataset_probs,
        strategy,
        args.train.seed,
        max_count=args.data.max_samples,
        dataset_split=args.data.dataset_split,
    )
    train_data = train_data.select(range(min(args.data.max_samples, len(train_data))))
    train_dataset = SFTDataset(
        train_data,
        tokenizer,
        args.data.max_len,
        strategy,
        pretrain_mode=args.model.pretrain_mode_enable,
        input_template=args.data.input_template,
        multiturn=args.data.multiturn,
    )
    train_dataloader = strategy.setup_dataloader(
        train_dataset,
        args.train.micro_batch_size,
        True,
        True,
        train_dataset.collate_fn,
        num_workers=args.data.dataloader_num_workers,
    )

    eval_dataloader = None
    if getattr(args.eval, "dataset", None):
        eval_data = blending_datasets(
            args.eval.dataset,
            None,
            strategy,
            dataset_split=args.eval.split,
        )
        eval_dataset = SFTDataset(
            eval_data,
            tokenizer,
            args.data.max_len,
            strategy,
            pretrain_mode=args.model.pretrain_mode_enable,
            input_template=args.data.input_template,
            multiturn=args.data.multiturn,
        )
        eval_dataloader = strategy.setup_dataloader(
            eval_dataset,
            args.train.micro_batch_size,
            True,
            False,
            eval_dataset.collate_fn,
            num_workers=args.data.dataloader_num_workers,
        )

    num_update_steps_per_epoch = len(train_dataset) // args.train.batch_size
    max_steps = math.ceil(args.train.max_epochs * num_update_steps_per_epoch)

    cfg = dict(
        optim=args.optim,
        muon=vars(args.muon),
        adam=vars(args.adam),
        lr_scheduler=args.lr_scheduler,
        lr_warmup_ratio=args.lr_warmup_ratio,
        min_lr_ratio=args.min_lr_ratio,
        max_norm=args.max_norm,
        scheduler_steps=max_steps,
    )
    model, optim, scheduler = strategy.prepare((model, cfg))

    consumed_samples = 0
    if args.ckpt.load_enable and os.path.exists(args.ckpt.path):
        load_path, states = strategy.load_ckpt(model.model, args.ckpt.path, optimizer=optim, scheduler=scheduler)
        if load_path is not None:
            consumed_samples = states["consumed_samples"]
            strategy.print(f"Loaded the checkpoint: {args.ckpt.path}, consumed_samples: {consumed_samples}")

    os.makedirs(args.ckpt.output_dir, exist_ok=True)

    trainer = SFTTrainer(
        model=model,
        strategy=strategy,
        optim=optim,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        scheduler=scheduler,
        max_norm=args.max_norm,
        pretrain_mode=args.model.pretrain_mode_enable,
        batch_size=args.train.batch_size,
        max_epochs=args.train.max_epochs,
        tokenizer=tokenizer,
        save_hf_ckpt=args.ckpt.save_hf,
        disable_ds_ckpt=args.ckpt.disable_ds,
    )

    trainer.fit(args, consumed_samples, num_update_steps_per_epoch)
    strategy.save_model(model, tokenizer, args.ckpt.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Checkpoint
    parser.add_argument("--ckpt.output_dir", type=str, default="./ckpt")
    parser.add_argument("--ckpt.save_steps", type=int, default=-1)
    parser.add_argument("--ckpt.save_hf", action="store_true", default=False)
    parser.add_argument("--ckpt.disable_ds", action="store_true", default=False)
    parser.add_argument("--logger.logging_steps", type=int, default=1)
    parser.add_argument("--eval.steps", type=int, default=-1)
    parser.add_argument("--ckpt.path", type=str, default="./ckpt/checkpoints_sft")
    parser.add_argument("--ckpt.max_num", type=int, default=3)
    parser.add_argument("--ckpt.max_mem", type=int, default=int(1e8))
    parser.add_argument("--ckpt.load_enable", action="store_true", default=False)

    # Training
    parser.add_argument("--train.micro_batch_size", type=int, default=8, help="batch size per GPU")
    parser.add_argument("--train.batch_size", type=int, default=128, help="Global training batch size")
    parser.add_argument("--train.seed", type=int, default=42)
    parser.add_argument(
        "--train.full_determinism_enable",
        action="store_true",
        default=False,
        help="Enable reproducible behavior during distributed training",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank from torchrun")

    # FSDP / Automodel backend
    parser.add_argument("--fsdp.tp_size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--fsdp.cp_size", type=int, default=1, help="Context parallel size (replaces ring-attn)")
    parser.add_argument("--fsdp.ep_size", type=int, default=1, help="Expert parallel size (MoE)")
    parser.add_argument("--fsdp.pp_size", type=int, default=1, help="Pipeline parallel size")
    parser.add_argument(
        "--fsdp.sequence_parallel",
        action="store_true",
        default=None,
        help="Sequence parallel within TP region. Default auto-on when --fsdp.tp_size>1.",
    )
    parser.add_argument("--fsdp.no_sequence_parallel", dest="fsdp.sequence_parallel", action="store_false")
    parser.add_argument("--fsdp.cpu_offload", action="store_true", default=False)
    parser.add_argument(
        "--fsdp.param_dtype", type=str, default="bf16", choices=["bf16", "fp16"], help="Model data type"
    )
    parser.add_argument(
        "--fsdp.attn_implementation",
        type=str,
        default="flash_attention_2",
        help="Attention implementation (e.g., eager, flash_attention_2, flash_attention_3, sdpa)",
    )
    parser.add_argument("--fsdp.use_liger_kernel", action="store_true", default=False, help="Enable Liger Kernel")
    parser.add_argument("--fsdp.packing_samples", action="store_true", default=False)
    parser.add_argument("--fsdp.load_in_4bit", action="store_true", default=False)
    parser.add_argument("--fsdp.lora.rank", type=int, default=0)
    parser.add_argument("--fsdp.lora.alpha", type=int, default=16)
    parser.add_argument("--fsdp.lora.target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--fsdp.lora.dropout", type=float, default=0)

    # Model
    parser.add_argument("--model.model_name_or_path", type=str, default=None)
    parser.add_argument("--model.gradient_checkpointing_enable", action="store_true", default=False)
    parser.add_argument("--model.gradient_checkpointing_reentrant", action="store_true", default=False)
    parser.add_argument("--model.pretrain_mode_enable", action="store_true", default=False, help="Use pretrain loss")
    parser.add_argument("--model.aux_loss_coef", type=float, default=0, help="MoE balancing loss")

    # Data
    parser.add_argument("--data.disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument(
        "--data.dataloader_num_workers", type=int, default=0, help="Number of dataloader workers for IO"
    )

    # SFT
    parser.add_argument("--train.max_epochs", type=int, default=2)

    # Optimizer + scheduler + grad clip
    parser.add_argument("--optim", type=str, default="adam", choices=["adam", "muon"])
    parser.add_argument("--muon.lr", type=float, default=0.02, help="LR for Muon 2D-weight group")
    parser.add_argument("--muon.momentum", type=float, default=0.95)
    parser.add_argument("--muon.ns_steps", type=int, default=5)
    parser.add_argument("--muon.nesterov", action="store_true", default=True)
    parser.add_argument("--muon.no_nesterov", dest="muon.nesterov", action="store_false")
    parser.add_argument("--adam.lr", type=float, default=5e-6)
    parser.add_argument("--adam.betas", type=float, nargs=2, default=(0.9, 0.95))
    parser.add_argument("--adam.eps", type=float, default=1e-8)
    parser.add_argument("--adam.weight_decay", type=float, default=0.0)
    parser.add_argument("--lr_scheduler", type=str, default="cosine_with_min_lr")
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.03)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--max_norm", type=float, default=1.0, help="Gradient clipping")

    # Custom dataset
    parser.add_argument("--data.dataset", type=str, default=None, help="Path to the training dataset")
    parser.add_argument(
        "--data.dataset_probs", type=str, default=None, help="Sampling probabilities for training datasets"
    )
    parser.add_argument("--eval.dataset", type=str, default=None, help="Path to the evaluation dataset")
    parser.add_argument("--data.dataset_split", type=str, default="train")
    parser.add_argument("--eval.split", type=str, default="train")
    parser.add_argument("--data.max_samples", type=int, default=1000000, help="Maximum number of samples to use")
    parser.add_argument("--data.multiturn", action="store_true", default=False, help="Use compacted multiturn dataset")

    parser.add_argument("--data.input_key", type=str, default="input", help="JSON dataset key")
    parser.add_argument("--data.output_key", type=str, default=None, help="JSON dataset key")
    parser.add_argument("--data.input_template", type=str, default="User: {}\nAssistant: ")
    parser.add_argument(
        "--data.apply_chat_template", action="store_true", default=False, help="Use HF tokenizer chat template"
    )
    parser.add_argument("--data.tokenizer_chat_template", type=str, default=None)
    parser.add_argument("--data.max_len", type=int, default=2048, help="Max tokens for the samples")

    # wandb parameters
    parser.add_argument("--logger.wandb.key", type=str, default=None)
    parser.add_argument("--logger.wandb.org", type=str, default=None)
    parser.add_argument("--logger.wandb.group", type=str, default=None)
    parser.add_argument("--logger.wandb.project", type=str, default="openrlhf_train_sft")
    parser.add_argument(
        "--logger.wandb.run_name",
        type=str,
        default="sft_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    # TensorBoard
    parser.add_argument("--logger.tensorboard_dir", type=str, default=None, help="TensorBoard logging path")

    # ModelScope
    parser.add_argument("--use_ms", action="store_true", default=False)

    args = parser.parse_args()
    from openrlhf.utils.config import hierarchize

    args = hierarchize(args)

    if args.data.multiturn:
        assert args.data.apply_chat_template, "apply_chat_template must be enabled when using multiturn format"

    if args.data.input_template and "{}" not in args.data.input_template:
        print("[Warning] '{}' not in args.data.input_template, set to None")
        args.data.input_template = None

    if args.data.input_template and "\\n" in args.data.input_template:
        print(
            "[Warning] input_template contains \\n characters instead of newline. "
            "You likely want to pass $'\\n' in Bash or \"`n\" in PowerShell."
        )

    if args.fsdp.packing_samples and args.fsdp.attn_implementation != "flash_attention_2":
        print(
            "[Warning] --fsdp.packing_samples no longer forces flash_attention_2. "
            "Automodel native models use THD packing with the selected backend; "
            "HF fallback models will disable packing unless flash_attention_2 + flash_attn is available."
        )

    if args.use_ms:
        from modelscope.utils.hf_util import patch_hub

        patch_hub()

    train(args)
