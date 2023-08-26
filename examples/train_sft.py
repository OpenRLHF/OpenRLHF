import argparse
import math
import os

from datetime import datetime
from datasets import load_dataset
from transformers.trainer import get_scheduler
from utils import blending_datasets, get_strategy, get_tokenizer

from openllama2.datasets import SFTDataset
from openllama2.models import Actor
from openllama2.trainer import SFTTrainer

# from openllama2.models.llama_flash_attn_monkey_patch import (
#     replace_llama_attn_with_flash_attn,
# )
# replace_llama_attn_with_flash_attn()


def train(args):
    # configure strategy
    strategy = get_strategy(args)

    # configure model
    # load huggingface model/config
    from_config = bool(args.load_model or args.load_checkpoint)
    model = Actor(args.pretrain, from_config)

    # load Pytorch model
    if args.load_model and not args.load_checkpoint:
        strategy.print("Load model: ", args.load_model)
        strategy.load_model(model, args.load_model)

    # lora
    if args.lora_rank > 0:
        model.lora_enable(args.lora_rank)

    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, model.model, 'right', strategy)

    # configure optimizer
    optim = strategy.create_optimizer(model, lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=args.l2)
    
    # prepare for data and dataset
    train_data, eval_data = blending_datasets(args.dataset, args.dataset_probs, strategy, args.seed)
    train_data = train_data.select(range(min(args.max_samples, len(train_data))))
    train_dataset = SFTDataset(train_data, tokenizer, args.max_len, strategy, pretrain_mode=args.pretrain_mode)
    eval_dataset = SFTDataset(eval_data, tokenizer, args.max_len, strategy, pretrain_mode=args.pretrain_mode)

    train_dataloader = strategy.setup_dataloader(train_dataset, args.micro_train_batch_size, 
                                                        True, True, train_dataset.collate_fn)
    eval_dataloader = strategy.setup_dataloader(eval_dataset, args.micro_train_batch_size,
                                                            True, False, eval_dataset.collate_fn)

    # scheduler
    num_update_steps_per_epoch = len(train_dataloader) // strategy.accumulated_gradient
    max_steps = math.ceil(args.max_epochs * num_update_steps_per_epoch)

    scheduler = get_scheduler("cosine",
                                optim,
                                num_warmup_steps=math.ceil(max_steps * 0.03),
                                num_training_steps=max_steps)    

    # prepare models
    (model, optim, scheduler) = strategy.prepare((model, optim, scheduler))

    # load checkpoint
    if args.load_checkpoint:
        strategy.print("Load checkpoint: ", args.save_path)
        # strategy.load_checkpoint(args.save_path + '/sft_model.pt')

    os.makedirs(args.save_path, exist_ok=True)

    # configure Trainer
    trainer = SFTTrainer(model=model,
                                 strategy=strategy,
                                 optim=optim,
                                 train_dataloader=train_dataloader,
                                 eval_dataloader=eval_dataloader,
                                 scheduler=scheduler,
                                 max_norm=args.max_norm,
                                 pretrain_mode=args.pretrain_mode,
                                 batch_size=args.train_batch_size,
                                 max_epochs=args.max_epochs,
                                 tokenizer=tokenizer,
                                 gradient_checkpointing=args.gradient_checkpointing,
                                 )

    trainer.fit(use_lora=args.lora_rank)
    
    # save model checkpoint after fitting on only rank0
    strategy.save_model(model, args.save_path + '/sft_model.pt', only_rank0=True)

    if args.save_hf_model:  
        strategy.save_hf_format(model, tokenizer, args.save_path + '/sft_hf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain', type=str, default='bigscience/bloomz-1b7')
    parser.add_argument('--dataset', type=str, default='Dahoas/full-hh-rlhf')
    parser.add_argument('--dataset_probs', type=str, default='1.0', help="sampling probs for datasets")
    parser.add_argument('--save_path', type=str, default='./ckpt')
    parser.add_argument('--max_epochs', type=int, default=2)
    parser.add_argument('--micro_train_batch_size', type=int, default=8)
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--max_samples', type=int, default=1000000)
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--max_norm', type=float, default=1.0)
    parser.add_argument('--l2', type=float, default=0)
    parser.add_argument('--load_model', type=str, default=None)
    parser.add_argument('--load_checkpoint', action='store_true', default=False)
    parser.add_argument('--pretrain_mode', action='store_true', default=False)
    parser.add_argument('--lora_rank', type=int, default=0, help="low-rank adaptation matrices rank")
    parser.add_argument('--gradient_checkpointing', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--local_rank', type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument('--zero_stage', type=int, default=2)
    parser.add_argument('--bf16', action='store_true', default=False)
    parser.add_argument('--learning_rate', type=float, default=2e-6)
    parser.add_argument('--zpg', type=int, default=8, help="ZeRO++ max partition size")
    parser.add_argument('--adam_offload', action="store_true", default=False)
    parser.add_argument('--save_hf_model', action='store_true', default=False)

    # wandb pamameters
    parser.add_argument('--use_wandb', type=str, default=None)
    parser.add_argument('--wandb_org', type=str, default=None)
    parser.add_argument('--wandb_group', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default="openllama2_train_sft")
    parser.add_argument('--wandb_run_name', type=str, default="sft_%s" % datetime.now().strftime('%m%dT%H:%M'))

    args = parser.parse_args()
    train(args)