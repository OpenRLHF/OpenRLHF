import argparse
import math
import os
from datetime import datetime

import torch
from transformers.trainer import get_scheduler

from openrlhf.models import Actor
from openrlhf.trainer import VLDPOTrainer
from openrlhf.utils import (
    get_strategy,
    get_tokenizer,
    blending_datasets,
    add_vision_args,
    add_extra_dataset_args,
    get_vision_processor,
)
from openrlhf.datasets.vl_reward_dataset import VLRewardDataset


def train(args):
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed()
    if torch.distributed.get_rank() == 0:
        print(f"Running args {args}")

    # configure model
    # load huggingface model
    model = Actor(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        ds_config=strategy.get_ds_train_config(is_actor=True),
        packing_samples=args.packing_samples,
        is_visual_model=args.is_visual_model,
    )

    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, model.model, "right", strategy, use_fast=not args.disable_fast_tokenizer)
    strategy.print(model)

    # configure processor
    vision_processor = get_vision_processor(args.pretrain, args.min_pixels, args.max_pixels)
    #TODO: support more model_arch
    # Other multimodal models are still undergoing adaptation. For the time being,
    # we are using this switch to impose restrictions and prevent any unforeseen issues.
    assert args.model_arch in ["qwen2_vl"
                               ], f"model arch {args.model_arch} is not supported currently"

    #TODO: support ring attention on vision model training
    assert strategy.ring_attn_group is None, f"Ring attention is not supported on vision models currently"

    # load weights for ref model
    ref_model = Actor(
        args.ref_pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        ds_config=strategy.get_ds_eval_config(offload=args.ref_offload),
        packing_samples=args.packing_samples,
        is_visual_model=args.is_visual_model,
    )
    if args.ref_offload:
        ref_model._offload = True

    # gradient_checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        )

    # configure optimizer
    optim = strategy.create_optimizer(model, lr=args.learning_rate, betas=args.adam_betas, weight_decay=args.l2)

    # repace datasets
    assert args.task_type == "dpo", f"the script is used for DPO training"

    #TODO: support ring attn and packing_samples
    assert not args.packing_samples, f"packing_samples is not supported currently"

    train_data, eval_data = blending_datasets(
        args.dataset,
        args.dataset_probs,
        strategy,
        args.seed,
        max_count=args.max_samples,
        train_split=args.train_split,
        eval_split=args.eval_split,
    )
    train_data = train_data.select(range(min(args.max_samples, len(train_data))))
    eval_data = eval_data.select(range(min(args.max_samples, len(eval_data))))

    train_dataset = VLRewardDataset(
        train_data,
        tokenizer,
        args.max_len,
        strategy,
        is_dpo=True,
        num_processors=args.processing_num_workers,
        processor=vision_processor,
        args=args,
    )
    eval_dataset = VLRewardDataset(
        eval_data,
        tokenizer,
        args.max_len,
        strategy,
        is_dpo=True,
        num_processors=args.processing_num_workers,
        processor=vision_processor,
        args=args,
    )

    # prepare dataloader
    train_dataloader = strategy.setup_dataloader(
        train_dataset,
        args.micro_train_batch_size,
        True,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )

    eval_dataloader = strategy.setup_dataloader(
        eval_dataset,
        args.micro_train_batch_size,
        True,
        shuffle=False,
        collate_fn=eval_dataset.collate_fn,
    )

    # scheduler
    num_update_steps_per_epoch = len(train_dataset) // args.train_batch_size
    max_steps = math.ceil(args.max_epochs * num_update_steps_per_epoch)

    scheduler = get_scheduler(
        args.lr_scheduler,
        optim,
        num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
        num_training_steps=max_steps,
        scheduler_specific_kwargs={"min_lr": args.learning_rate * 0.1},
    )

    # strategy prepare
    ((model, optim, scheduler), ref_model) = strategy.prepare((model, optim, scheduler), ref_model)

    # load checkpoint
    consumed_samples = 0
    if args.load_checkpoint and os.path.exists(args.ckpt_path):
        _, states = strategy.load_ckpt(model.model, args.ckpt_path)
        consumed_samples = states["consumed_samples"]
        strategy.print(f"Loaded the checkpoint: {args.ckpt_path}, consumed_samples: {consumed_samples}")

    os.makedirs(args.save_path, exist_ok=True)

    # batch_size here is expected to be C(k,2), k means # response of each prompt
    # be limited with the format of dataset 'Dahoas/rm-static', we'd better use batch_size as 1
    trainer = VLDPOTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        strategy=strategy,
        optim=optim,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        scheduler=scheduler,
        max_norm=args.max_norm,
        beta=args.beta,
        max_epochs=args.max_epochs,
    )

    trainer.fit(args, consumed_samples, num_update_steps_per_epoch)

    # save model checkpoint after fitting on only rank0
    strategy.save_model(model, tokenizer, args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Checkpoints
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_dpo")
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1e8)

    # DeepSpeed
    parser.add_argument("--micro_train_batch_size", type=int, default=8, help="batch size per GPU")
    parser.add_argument("--train_batch_size", type=int, default=128, help="Global training batch size")
    parser.add_argument("--load_checkpoint", action="store_true", default=False)
    parser.add_argument("--max_norm", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage")
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    parser.add_argument("--ref_offload", action="store_true", default=False)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.03)
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False, help="Offload Adam Optimizer")
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")
    parser.add_argument("--grad_accum_dtype", type=str, default=None, help="Adam grad accum data type")
    parser.add_argument("--overlap_comm", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False)

    # DPO
    parser.add_argument("--lr_scheduler", type=str, default="cosine_with_min_lr")
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--l2", type=float, default=0.0, help="weight decay loss")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--ipo", action="store_true", default=False)  # IPO https://arxiv.org/pdf/2310.12036v2.pdf
    parser.add_argument("--label_smoothing", type=float, default=0.0)  # cDPO https://arxiv.org/pdf/2305.18290.pdf
    parser.add_argument("--aux_loss_coef", type=float, default=0, help="MoE balancing loss")
    parser.add_argument(
        "--nll_loss_coef", type=float, default=0, help="Regularization with NLL loss, see LLama 3.1 tech report."
    )
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95), help="Betas for Adam optimizer")

    # Context Parallel
    parser.add_argument("--ring_attn_size", type=int, default=1, help="Ring attention group size")
    parser.add_argument(
        "--ring_head_stride",
        type=int,
        default=1,
        help="the number of heads to do ring attention each time. "
        "It should be a divisor of the number of heads. "
        "A larger value may results in faster training but will consume more memory.",
    )

    # LoRA
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--lora_dropout", type=float, default=0)

    # packing samples using Flash Attention2
    parser.add_argument("--packing_samples", action="store_true", default=False)

    # Custom dataset
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--ref_pretrain", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--dataset_probs", type=str, default="1.0", help="sampling probs for datasets")
    parser.add_argument("--train_split", type=str, default="train", help="train split of the HF dataset")
    parser.add_argument("--eval_split", type=str, default="test", help="test split of the dataset")

    parser.add_argument("--prompt_key", type=str, default=None)
    parser.add_argument("--chosen_key", type=str, default="chosen")
    parser.add_argument("--rejected_key", type=str, default="rejected")
    parser.add_argument("--input_template", type=str, default=None)
    parser.add_argument(
        "--apply_chat_template", action="store_true", default=False, help="Use HF tokenizer chat template"
    )
    parser.add_argument("--max_samples", type=int, default=1e8, help="Max number of samples")
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--min_pixels", type=int, default=3136, help="min width*height for image")
    parser.add_argument("--max_pixels", type=int, default=12845056, help="max width*height for image")

    # wandb parameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_dpo")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="exp_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    # TensorBoard parameters
    parser.add_argument("--use_tensorboard", type=str, default=None, help="TensorBoard logging path")

    parser = add_vision_args(parser)
    parser = add_extra_dataset_args(parser)
    args = parser.parse_args()

    if args.ref_pretrain is None or args.ref_pretrain == "":
        args.ref_pretrain = args.pretrain

    if args.input_template and "{}" not in args.input_template:
        print("[Warning] {} not in args.input_template, set to None")
        args.input_template = None

    if args.input_template and "\\n" in args.input_template:
        print(
            "[Warning] input_template contains \\n chracters instead of newline. "
            "You likely want to pass $'\\n' in Bash or \"`n\" in PowerShell."
        )

    if args.packing_samples and not args.flash_attn:
        print("[Warning] Please --flash_attn to accelerate when --packing_samples is enabled.")
        args.flash_attn = True

    if args.ring_attn_size > 1:
        assert args.packing_samples, "packing_samples must be enabled when using ring attention"

    train(args)
