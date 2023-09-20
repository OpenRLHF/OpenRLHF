import argparse
import math
import os
from collections import OrderedDict
from datetime import datetime

from transformers.trainer import get_scheduler

from openllama2.datasets import RewardDataset
from openllama2.models import RewardModel
from openllama2.trainer import RewardModelTrainer
from openllama2.utils import blending_datasets, get_strategy, get_tokenizer


def train(args):
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed()

    # configure flash attention
    if args.flash_attn:
        from openllama2.models.llama2_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

        replace_llama_attn_with_flash_attn()

    # configure model
    # load huggingface model/config
    from_config = bool(args.load_model or args.load_checkpoint)
    model = RewardModel(args.pretrain, from_config)

    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, model.model, "left", strategy)

    strategy.print(model)

    # load SFT model
    if args.load_model and not args.load_checkpoint:

        def key_replace_fn(states_dict):
            new_state_dict = OrderedDict()
            for k, v in states_dict.items():
                new_state_dict[k.replace("transformer.", "model.")] = v
            return new_state_dict

        strategy.load_model(model, args.load_model, strict=False, key_replace_fn=key_replace_fn)
        strategy.print("Load model: ", args.load_model)

    # lora
    if args.lora_rank > 0:
        model.lora_enable(args.lora_rank)

    # configure optimizer
    optim = strategy.create_optimizer(model, lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=args.l2)

    # prepare for data and dataset
    train_data, eval_data = blending_datasets(
        args.dataset,
        args.dataset_probs,
        strategy,
        args.seed,
        max_count=2000000,
        stopping_strategy="all_exhausted",
    )
    train_dataset = RewardDataset(train_data, tokenizer, args.max_len, strategy) if not args.only_evaluate else None
    eval_dataset = RewardDataset(eval_data, tokenizer, args.max_len, strategy)

    train_dataloader = (
        strategy.setup_dataloader(
            train_dataset,
            args.micro_train_batch_size,
            True,
            True,
            train_dataset.collate_fn,
        )
        if not args.only_evaluate
        else None
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
    (model, optim, scheduler) = strategy.prepare((model, optim, scheduler))

    if args.load_checkpoint:
        strategy.print("Load checkpoint: ", args.save_path)
        # strategy.load_checkpoint(args.save_path + '/rm_model.pt')

    os.makedirs(args.save_path, exist_ok=True)

    # batch_size here is expected to be C(k,2), k means # response of each prompt
    # be limited with the format of dataset 'Dahoas/rm-static', we'd better use batch_size as 1
    trainer = RewardModelTrainer(
        model=model,
        strategy=strategy,
        optim=optim,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        scheduler=scheduler,
        max_norm=args.max_norm,
        batch_size=args.train_batch_size,
        max_epochs=args.max_epochs,
        gradient_checkpointing=args.gradient_checkpointing,
        only_evaluate=args.only_evaluate,
        loss=args.loss,
    )

    trainer.fit(use_lora=args.lora_rank)

    if not args.only_evaluate:
        # save model checkpoint after fitting on only rank0
        strategy.save_model(model, args.save_path + "/rm_model.pt", only_rank0=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", type=str, default="bigscience/bloomz-1b7")
    # parser.add_argument('--dataset', type=str, default='Anthropic/hh-rlhf')
    parser.add_argument("--dataset", type=str, default="Dahoas/full-hh-rlhf")
    parser.add_argument("--dataset_probs", type=str, default="1.0", help="sampling probs for datasets")
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--micro_train_batch_size", type=int, default=8)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--only_evaluate", action="store_true", default=False)
    parser.add_argument("--load_checkpoint", action="store_true", default=False)
    parser.add_argument("--load_model", type=str, default=None)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--loss", type=str, default="sigmoid")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora_rank", type=int, default=0, help="low-rank adaptation matrices rank")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--zpg", type=int, default=8, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False)
    parser.add_argument("--flash_attn", action="store_true", default=False)

    # wandb pamameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openllama2_train_rm")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="rm_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    args = parser.parse_args()
    train(args)
