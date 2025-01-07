import argparse
import math
import os
from datetime import datetime

from transformers.trainer import get_scheduler

from openrlhf.datasets import ProcessRewardDataset
from openrlhf.models import Actor
from openrlhf.trainer import ProcessRewardModelTrainer
from openrlhf.utils import blending_datasets, get_strategy, get_tokenizer


def train(args):
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
        lora_dropout=args.lora_dropout,
        ds_config=strategy.get_ds_train_config(is_actor=True),
        packing_samples=args.packing_samples,
    )
    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, model.model, "right", strategy, use_fast=not args.disable_fast_tokenizer)
    if args.specialize_reward_tokens:
        additional_special_tokens = [
            args.placeholder_token_in_tokenizer,
            *args.reward_tokens_in_tokenizer,
        ]
        old_vocab_size = len(tokenizer)
        tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})
        new_vocab_size = len(tokenizer)
        assert new_vocab_size == old_vocab_size, (
            f"placeholder token {repr(args.placeholder_token_in_tokenizer)} does not exist in tokenizer.\n"
            "Please consider to choose reserved trained tokens as placeholder/rewards tokens. "
            "Or please disable --specialize_reward_tokens option."
        )
        strategy.print((
            f"placeholder token {repr(args.placeholder_token_in_tokenizer)} "
            f"and reward tokens {repr(args.reward_tokens_in_tokenizer)} "
            "are added to tokenizer as additional_special_tokens"
        ))
        for token in additional_special_tokens:
            strategy.print((
                f"{repr(token)} -> {tokenizer(token)['input_ids']}, "
                f"{repr(' '+ token)} -> {tokenizer(' ' + token)['input_ids']}"
            ))
    strategy.print(model)

    # gradient_checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        )

    # configure optimizer
    optim = strategy.create_optimizer(model, lr=args.learning_rate, betas=args.adam_betas, weight_decay=args.l2)

    # prepare for data and dataset
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

    def preprocessing(sample):
        # placeholder token is replaced to reserved token in tokenizer to avoid tokenization issue.
        # see https://github.com/OpenRLHF/OpenRLHF/issues/645
        if args.input_key in sample and args.placeholder_token != args.placeholder_token_in_tokenizer:
            sample[args.input_key] = sample[args.input_key].replace(
                args.placeholder_token,
                args.placeholder_token_in_tokenizer
            )
        if args.label_key in sample and args.reward_tokens != args.reward_tokens_in_tokenizer:
            label_list = sample[args.label_key]
            if label_list is not None and len(label_list) > 0 and isinstance(label_list[0], str):
                new_label_list = []
                for label in label_list:
                    for reward_token, reward_token_in_tokenizer in zip(args.reward_tokens, args.reward_tokens_in_tokenizer):
                        label = label.replace(reward_token, reward_token_in_tokenizer)
                        new_label_list.append(label)
                sample[args.label_key] = label_list
        return sample

    train_data = train_data.map(preprocessing)
    eval_data = eval_data.map(preprocessing)

    def print_sample(sample) -> None:
        input_text = sample[args.input_key]
        labels = sample[args.label_key]
        strategy.print("inputs:\n{}".format(tokenizer.decode(tokenizer.encode(input_text), skip_special_tokens=False)))
        strategy.print("input_ids:\n{}".format(tokenizer.encode(input_text)))
        strategy.print("labels:\n{}".format(labels))
        if isinstance(labels[0], str):
            strategy.print("label_ids:\n{}".format([tokenizer.encode(i, add_special_tokens=False)[-1] for i in labels]))

    print_sample(train_data[0])

    train_dataset = ProcessRewardDataset(train_data, tokenizer, args.max_len, strategy)
    eval_dataset = ProcessRewardDataset(eval_data, tokenizer, args.max_len, strategy)

    # prepare dataloader
    train_dataloader = strategy.setup_dataloader(
        train_dataset,
        args.micro_train_batch_size,
        True,
        True,
        train_dataset.packing_collate_fn if args.packing_samples else train_dataset.collate_fn,
    )
    eval_dataloader = strategy.setup_dataloader(
        eval_dataset,
        args.micro_train_batch_size,
        True,
        False,
        eval_dataset.packing_collate_fn if args.packing_samples else eval_dataset.collate_fn,
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

    # prepare models
    (model, optim, scheduler) = strategy.prepare((model, optim, scheduler))

    # load checkpoint
    consumed_samples = 0
    if args.load_checkpoint and os.path.exists(args.ckpt_path):
        _, states = strategy.load_ckpt(model.model, args.ckpt_path)
        consumed_samples = states["consumed_samples"]
        strategy.print(f"Loaded the checkpoint: {args.ckpt_path}, consumed_samples: {consumed_samples}")

    os.makedirs(args.save_path, exist_ok=True)

    # configure Trainer
    trainer = ProcessRewardModelTrainer(
        model=model,
        strategy=strategy,
        optim=optim,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        scheduler=scheduler,
        max_norm=args.max_norm,
        batch_size=args.train_batch_size,
        max_epochs=args.max_epochs,
        tokenizer=tokenizer,
    )

    trainer.fit(args, consumed_samples, num_update_steps_per_epoch)

    # save model checkpoint after fitting on only rank0
    strategy.save_model(model, tokenizer, args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Checkpoint
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_prm")
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1e8)
    parser.add_argument("--load_checkpoint", action="store_true", default=False)

    # DeepSpeed
    parser.add_argument("--max_norm", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage")
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False, help="Offload Adam Optimizer")
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")
    parser.add_argument("--grad_accum_dtype", type=str, default=None, help="Adam grad accum data type")
    parser.add_argument("--overlap_comm", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)

    # LoRA
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")

    # PRM training
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--aux_loss_coef", type=float, default=0, help="MoE balancing loss")
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.03)
    parser.add_argument("--micro_train_batch_size", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=128, help="Global training batch size")
    parser.add_argument("--lr_scheduler", type=str, default="cosine_with_min_lr")
    parser.add_argument("--l2", type=float, default=0.0, help="weight decay loss")
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95), help="Betas for Adam optimizer")
    parser.add_argument("--placeholder_token", type=str, default=None, help="placeholder token in dataset")
    parser.add_argument("--reward_tokens", type=str, nargs="*", default=None)
    parser.add_argument("--specialize_reward_tokens", action="store_true", default=False)
    parser.add_argument("--placeholder_token_in_tokenizer", type=str, default=None,
                        help="placeholder_token in dataset will be repalced to reserved token in tokenizer when preprocessing.")
    parser.add_argument("--reward_tokens_in_tokenizer", type=str, nargs="*", default=None,
                        help="reward_tokens in dataset will be repalced to reserved tokens in tokenizer when preprocessing.")

    # packing samples using Flash Attention2
    parser.add_argument("--packing_samples", action="store_true", default=False)

    # custom dataset
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--dataset_probs", type=str, default="1.0", help="sampling probs for datasets")
    parser.add_argument("--train_split", type=str, default="train", help="train split of the HF dataset")
    parser.add_argument("--eval_split", type=str, default="test", help="test split of the dataset")

    parser.add_argument("--input_key", type=str, default="input", help="JSON dataset key")
    parser.add_argument("--label_key", type=str, default="label", help="JSON dataset key")
    parser.add_argument("--max_samples", type=int, default=1e8, help="Max number of samples")
    parser.add_argument("--max_len", type=int, default=2048, help="Max tokens for the samples")

    # wandb parameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_prm")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="prm_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    args = parser.parse_args()

    # Add positive token and negative token to reward_tokens and remove duplicates
    if args.reward_tokens is not None:
        print(
            "If you are running with soft labels (float values), "
            f"the first token in reward_tokens ({args.reward_tokens[0]}) should be the positive token "
            "and the second token should be the negative token."
        )

    if args.placeholder_token_in_tokenizer is None:
        args.placeholder_token_in_tokenizer = args.placeholder_token
        print(
            "Option '--placeholder_token_in_tokenizer' is None. "
            f"placeholder_token_in_tokenizer is set to the placeholder token {repr(args.placeholder_token)} in dataset by default."
        )
    if args.reward_tokens_in_tokenizer is None:
        args.reward_tokens_in_tokenizer = args.reward_tokens
        print(
            "Option '--reward_tokens_in_tokenizer' is None. "
            f"reward_tokens_in_tokenizer is set to the reward tokens {repr(args.reward_tokens)} in dataset by default."
        )
    train(args)
