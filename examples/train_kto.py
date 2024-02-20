import argparse
import math
import os
from collections import OrderedDict
from datetime import datetime

import torch.distributed as dist
from transformers.trainer import get_scheduler

from openrlhf.datasets import (
    DistributedVanillaKTOSampler,
    RewardDataset,
    UnpairedPreferenceDataset,
    UnpairedRewardDataset,
)
from openrlhf.models import Actor
from openrlhf.trainer import KTOTrainer
from openrlhf.utils import blending_datasets, get_strategy, get_tokenizer


def train(args):
    if args.unpaired_preference:
        assert args.vanilla_loss is False, "vanilla_loss is not supported for unpaired_preference"

    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed()

    # configure model
    # load huggingface model
    model = Actor(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        ds_config=strategy.get_ds_train_config(is_actor=True),
    )

    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, model.model, "right", strategy)
    strategy.print(model)

    # load weights for ref model
    ref_model = Actor(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        ds_config=strategy.get_ds_eval_config(offload=args.ref_offload),
    )
    if args.ref_offload:
        ref_model._offload = True
    get_tokenizer(args.pretrain, ref_model.model, "right", strategy)

    # gradient_checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        )

    # configure optimizer
    optim = strategy.create_optimizer(model, lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=args.l2)

    # prepare for data and dataset
    train_data, eval_data = blending_datasets(
        args.dataset,
        args.dataset_probs,
        strategy,
        args.seed,
        max_count=args.max_samples,
        stopping_strategy="all_exhausted",
    )
    train_data = train_data.select(range(min(args.max_samples, len(train_data))))
    eval_data = eval_data.select(range(min(args.max_samples, len(eval_data))))

    if not args.unpaired_preference:
        train_dataset = RewardDataset(
            train_data, tokenizer, args.max_len, strategy, input_template=args.input_template
        )
        eval_dataset = RewardDataset(eval_data, tokenizer, args.max_len, strategy, input_template=args.input_template)
        train_dataset = UnpairedRewardDataset(train_dataset, vanilla_loss=args.vanilla_loss)
        eval_dataset = UnpairedRewardDataset(eval_dataset, vanilla_loss=args.vanilla_loss)

        train_sampler = DistributedVanillaKTOSampler(
            train_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=False,
            seed=args.seed,
            drop_last=False,
        )
        train_dataloader = strategy.setup_dataloader(
            train_dataset,
            args.micro_train_batch_size,
            True,
            True,
            train_dataset.collate_fn,
            sampler=train_sampler,
        )

        eval_sampler = DistributedVanillaKTOSampler(
            eval_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=False,
            seed=args.seed,
            drop_last=False,
        )
        eval_dataloader = strategy.setup_dataloader(
            eval_dataset,
            args.micro_train_batch_size,
            True,
            False,
            eval_dataset.collate_fn,
            sampler=eval_sampler,
        )
    else:
        train_dataset = UnpairedPreferenceDataset(
            train_data, tokenizer, args.max_len, strategy, input_template=args.input_template
        )
        eval_dataset = UnpairedPreferenceDataset(
            eval_data, tokenizer, args.max_len, strategy, input_template=args.input_template
        )
        train_dataloader = strategy.setup_dataloader(
            train_dataset,
            args.micro_train_batch_size,
            True,
            True,
            train_dataset.collate_fn,
        )
        eval_dataloader = strategy.setup_dataloader(
            eval_dataset, args.micro_train_batch_size, True, False, eval_dataset.collate_fn
        )

    # scheduler
    num_update_steps_per_epoch = len(train_dataloader) * args.max_epochs // strategy.accumulated_gradient
    max_steps = math.ceil(args.max_epochs * num_update_steps_per_epoch)

    scheduler = get_scheduler(
        "cosine",
        optim,
        num_warmup_steps=math.ceil(max_steps * 0.03),
        num_training_steps=max_steps,
    )

    # strategy prepare
    ((model, optim, scheduler), ref_model) = strategy.prepare((model, optim, scheduler), ref_model)

    os.makedirs(args.save_path, exist_ok=True)

    trainer = KTOTrainer(
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
        vanilla_loss=args.vanilla_loss,
    )
    trainer.fit(args)

    # save model checkpoint after fitting on only rank0
    strategy.save_model(model, tokenizer, args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", type=str, default="bigscience/bloomz-1b7")
    parser.add_argument("--dataset", type=str, default="Dahoas/full-hh-rlhf")
    parser.add_argument("--dataset_probs", type=str, default="1.0", help="sampling probs for datasets")
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_kto")
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1000)  # 1000GB
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--micro_train_batch_size", type=int, default=8)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--load_checkpoint", action="store_true", default=False)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--vanilla_loss",
        action="store_true",
        default=False,
        help="make sure there are as many positive and negative samples in the batch",
    )
    parser.add_argument(
        "--unpaired_preference",
        action="store_true",
        default=False,
        help="dataset is the format of unpaired preference",
    )

    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--ref_offload", action="store_true", default=False)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False)
    parser.add_argument("--flash_attn", action="store_true", default=False)
    parser.add_argument("--max_samples", type=int, default=1000000)
    parser.add_argument("--aux_loss_coef", type=float, default=0)
    parser.add_argument("--grad_accum_dtype", type=str, default=None)
    parser.add_argument("--disable_trace_cache", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=list, default=None)
    parser.add_argument("--input_template", type=str, default="Human: {}\nAssistant: ")
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true")

    # custom dataset key name
    # pair-wise
    parser.add_argument("--prompt_key", type=str, default=None)
    parser.add_argument("--chosen_key", type=str, default=None)
    parser.add_argument("--rejected_key", type=str, default=None)
    # unpair
    parser.add_argument("--output_key", type=str, default=None)
    parser.add_argument("--label_key", type=str, default=None)

    # wandb pamameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_dpo")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="rm_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    args = parser.parse_args()
    train(args)
