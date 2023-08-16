import argparse
import itertools
import math
import os
from copy import deepcopy

import torch
import torch.distributed as dist
from datasets import load_dataset
from transformers.trainer import get_scheduler
from utils import blending_datasets, get_strategy, get_tokenizer

from openllama2.datasets import PromptDataset, SFTDataset
from openllama2.models import Actor, Critic, RewardModel
from openllama2.trainer import PPOTrainer

# from openllama2.models.llama_flash_attn_monkey_patch import (
#     replace_llama_attn_with_flash_attn,
# )
# replace_llama_attn_with_flash_attn()


def train(args):
    # configure strategy
    strategy = get_strategy(args)

    # configure model
    # load huggingface model/config
    actor_from_config = bool(args.sft_model_path or args.load_checkpoint)
    reward_from_config = bool(args.reward_model_path)

    actor = Actor(args.pretrain, actor_from_config)
    critic = Critic(args.critic_pretrain, True, args.normalize_reward)
    reward_model = RewardModel(args.critic_pretrain, reward_from_config, args.normalize_reward)

    # load PyTorch model
    if args.sft_model_path:
        strategy.load_model(actor, args.sft_model_path)
    if args.reward_model_path:
        strategy.load_model(reward_model, args.reward_model_path)
    
    # copy weights for reference actor/ema actor/critic
    initial_model = deepcopy(actor)
    critic.model = deepcopy(reward_model.model)
    critic.value_head = deepcopy(reward_model.value_head)

    strategy.print("reward normalization status: {}".format(args.normalize_reward))
    strategy.print("mean: {}, std {}".format(reward_model.mean, reward_model.std))
    critic.mean = deepcopy(reward_model.mean)
    critic.std = deepcopy(reward_model.std)

    # lora
    if args.lora_rank > 0:
        strategy.print("lora_enable")
        actor.lora_enable(args.lora_rank)
        critic.lora_enable(args.lora_rank)

    if args.enable_ema:
        ema_model = deepcopy(actor)
    else:
        ema_model = None

    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, actor.model, 'left', strategy)

    # configure optimizer
    actor_optim = strategy.create_optimizer(actor, lr=args.actor_learning_rate, betas=(0.9, 0.95), weight_decay=args.l2)
    critic_optim = strategy.create_optimizer(critic, lr=args.critic_learning_rate, betas=(0.9, 0.95), weight_decay=args.l2)

    # prepare datasets
    prompts_data = blending_datasets(args.prompt_data, args.prompt_data_probs, strategy, args.seed, max_count=100000, return_eval=False)
    prompts_data = prompts_data.select(range(min(args.max_samples, len(prompts_data))))
    prompts_dataset = PromptDataset(prompts_data, strategy)
    prompts_dataloader = strategy.setup_dataloader(prompts_dataset, args.micro_rollout_batch_size, True, True)

    if args.pretrain_data:
        pretrain_data = blending_datasets(args.pretrain_data, args.pretrain_data_probs, strategy, args.seed, return_eval=False)
        pretrain_max_len =  args.max_len if args.max_len else args.prompt_max_len + args.generate_max_len
        pretrain_dataset = SFTDataset(pretrain_data.select(range(min(len(pretrain_data), args.max_epochs * len(prompts_dataset)))), 
                    tokenizer, pretrain_max_len, strategy, pretrain_mode=True)
        pretrain_dataloader = itertools.cycle(iter(strategy.setup_dataloader(pretrain_dataset, args.micro_train_batch_size, 
                                                    True, True, pretrain_dataset.collate_fn)))
    else:
        pretrain_dataloader = None

    # configure scheduler
    num_update_steps_per_episodes = len(prompts_dataloader) * args.max_epochs // strategy.accumulated_gradient
    max_steps = math.ceil(args.num_episodes * num_update_steps_per_episodes)

    actor_scheduler = get_scheduler("constant_with_warmup",
                                actor_optim,
                                num_warmup_steps=math.ceil(max_steps * 0.03),
                                num_training_steps=max_steps)    

    critic_scheduler = get_scheduler("constant_with_warmup",
                                critic_optim,
                                num_warmup_steps=math.ceil(max_steps * 0.03),
                                num_training_steps=max_steps)    

    # prepare models/optimizers...
    (actor, actor_optim, actor_scheduler), (critic, critic_optim, critic_scheduler), reward_model, initial_model \
    = strategy.prepare((actor, actor_optim, actor_scheduler), 
                       (critic, critic_optim, critic_scheduler), reward_model, initial_model, is_rlhf=True)

    if ema_model:
        ema_model.is_ema = True
        ema_model = strategy.prepare(ema_model, is_rlhf=True)
        del ema_model.is_ema

    # load checkpoint
    if args.load_checkpoint:
        strategy.print("Load checkpoint: ", args.save_path)
        #strategy.load_checkpoint(args.save_path + '/ppo_model.pt')

    os.makedirs(args.save_path, exist_ok=True)

    # configure Trainer
    trainer = PPOTrainer(
        strategy,
        actor,
        critic,
        reward_model,
        initial_model,
        ema_model,
        actor_optim,
        critic_optim,
        actor_scheduler,
        critic_scheduler,
        max_epochs=args.max_epochs,
        micro_train_batch_size=args.micro_train_batch_size,
        micro_rollout_batch_size=args.micro_rollout_batch_size,
        gradient_checkpointing=args.gradient_checkpointing,
        tokenizer=tokenizer,
        prompt_max_len=args.prompt_max_len,
        value_clip=args.value_clip,
        eps_clip=args.eps_clip,
        gamma=args.gamma,
        lambd=args.lambd,
        init_kl_coef=args.init_kl_coef,
        kl_target=args.kl_target,
        ema_beta=0.992,
        ptx_coef=args.ptx_coef,
        max_norm=args.max_norm,
        # fro GPT generation
        do_samples=True,
        max_new_tokens=args.generate_max_len,
        max_length=args.max_len,
        temperature=1,
        top_k=args.top_k,
        top_p=args.top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    trainer.fit(prompts_dataloader,
                pretrain_dataloader,
                num_episodes=args.num_episodes,
                rollout_batch_size=args.rollout_batch_size)
    
    # save model checkpoint after fitting on only rank0
    strategy.save_model(ema_model if args.enable_ema else actor, 
                        args.save_path + '/ppo_model.pt', 
                        only_rank0=True)

    if args.save_hf_model:  
        strategy.save_hf_format(ema_model if args.enable_ema else actor, 
                                tokenizer, args.save_path + '/ppo_hf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_data', type=str, default=None)
    parser.add_argument('--prompt_data_probs', type=str, default="1.0", help="sampling probs for datasets")
    parser.add_argument('--pretrain_data', type=str, default=None)
    parser.add_argument('--pretrain_data_probs', type=str, default="1.0", help="sampling probs for datasets")
    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--critic_pretrain', type=str, default=None)
    parser.add_argument('--reward_model_path', type=str, default=None)
    parser.add_argument('--sft_model_path', type=str, default=None)
    parser.add_argument('--save_path', type=str, default='./ckpt')
    parser.add_argument('--num_episodes', type=int, default=1)
    parser.add_argument('--rollout_batch_size', type=int, default=512)
    parser.add_argument('--micro_rollout_batch_size', type=int, default=8)
    parser.add_argument('--max_epochs', type=int, default=1)
    parser.add_argument('--prompt_max_len', type=int, default=1024)
    parser.add_argument('--generate_max_len', type=int, default=1024)
    parser.add_argument('--max_len', type=int, default=None)
    parser.add_argument('--max_samples', type=int, default=100000)
    parser.add_argument('--max_norm', type=float, default=1.0)
    parser.add_argument('--l2', type=float, default=0.)
    parser.add_argument('--ptx_coef', type=float, default=0.05)
    parser.add_argument('--eps_clip', type=float, default=0.2)
    parser.add_argument('--value_clip', type=float, default=0.2)
    parser.add_argument('--lambd', type=float, default=0.95)
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--micro_train_batch_size', type=int, default=4)
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--load_checkpoint', action='store_true', default=False)
    parser.add_argument('--normalize_reward', action='store_true', default=False)
    parser.add_argument('--top_k', type=int, default=None)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lora_rank', type=int, default=0, help="low-rank adaptation matrices rank")
    parser.add_argument('--local_rank', type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument('--zero_stage', type=int, default=2)
    parser.add_argument('--inference_tp_size', type=int, default=1)
    parser.add_argument('--gradient_checkpointing', action='store_true', default=False)
    parser.add_argument('--bf16', action='store_true', default=False)
    parser.add_argument('--actor_learning_rate', type=float, default=1e-6)
    parser.add_argument('--critic_learning_rate', type=float, default=9e-6)
    parser.add_argument('--kl_target', type=float, default=None)
    parser.add_argument('--init_kl_coef', type=float, default=0.02)
    ## Make EMA as an optional feature
    parser.add_argument('--enable_ema',
                        action='store_true',
                        help='Enable EMA checkpoint for the model.')
    parser.add_argument('--zpg', type=int, default=8, help="ZeRO++ max partition size")
    parser.add_argument('--adam_offload', action="store_true", default=False)
    parser.add_argument('--save_hf_model', action='store_true', default=False)
    args = parser.parse_args()
    train(args)
