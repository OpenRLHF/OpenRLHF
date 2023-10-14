import argparse
import os
from datetime import timedelta

import jsonlines
import torch
from torch import distributed as dist
from tqdm import tqdm

from openllama2.datasets import PromptDataset, SFTDataset
from openllama2.models import Actor, RewardModel
from openllama2.utils import blending_datasets, get_strategy, get_tokenizer


def batch_generate(args):
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed(timeout=timedelta(minutes=99999))

    # configure model
    from_config = bool(args.load_model)
    model = Actor(args.pretrain, from_config, use_flash_attention_2=args.flash_attn)

    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, model.model, "left", strategy)

    # load Pytorch model
    if args.load_model:
        strategy.print("Load model: ", args.load_model)
        strategy.load_model(model, args.load_model)

    # prepare models
    model = strategy.prepare(model)
    model.eval()

    # tokenizer
    def tokenize_fn(texts):
        batch = tokenizer(
            texts,
            return_tensors="pt",
            max_length=args.prompt_max_len,
            padding=True,
            truncation=True,
        )
        return {k: v.to(torch.cuda.current_device()) for k, v in batch.items()}

    prompts_data = blending_datasets(
        args.dataset,
        args.dataset_probs,
        strategy,
        args.seed,
        return_eval=False,
    )
    prompts_data = prompts_data.select(range(min(args.max_samples, len(prompts_data))))
    prompts_dataset = PromptDataset(prompts_data, strategy)
    prompts_dataloader = strategy.setup_dataloader(
        prompts_dataset, args.micro_batch_size, True, False, drop_last=False
    )
    pbar = tqdm(
        prompts_dataloader,
        disable=not strategy.is_rank_0(),
    )

    output_dataset = []
    for prompts in pbar:
        inputs = tokenize_fn(prompts)
        outputs = model.model.generate(
            **inputs,
            use_cache=True,
            max_length=args.max_len,
            do_sample=not args.greedy_sampling,
            top_p=args.top_p,
            early_stopping=True,
            num_beams=1,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for prompt, output in zip(prompts, outputs):
            output = output[len(prompt) :]
            output_dataset.append({"input": prompt, "output": output})

        with jsonlines.open(args.output_path + str(strategy.get_rank()), mode="w") as writer:
            writer.write_all(output_dataset)

    # wait unitl all processes generate done
    dist.barrier()

    # concate multiple output files in rank 0
    if strategy.is_rank_0():
        output_dataset = []
        world_size = dist.get_world_size()
        files = [args.output_path + str(rank) for rank in range(world_size)]
        for file in files:
            with jsonlines.open(file, mode="r") as reader:
                for obj in reader:
                    output_dataset.append(obj)
            os.remove(file)

        with jsonlines.open(args.output_path, mode="w") as writer:
            writer.write_all(output_dataset)


def batch_rm_inference(args):
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed(timeout=timedelta(minutes=99999))

    # configure model
    # load huggingface model/config
    from_config = bool(args.load_model)
    model = RewardModel(args.pretrain, from_config, use_flash_attention_2=args.flash_attn)

    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, model.model, "left", strategy)

    # load RM
    if args.load_model:
        strategy.load_model(model, args.load_model)
        strategy.print("Load model: ", args.load_model)

    # prepare models
    model = strategy.prepare(model)
    model.eval()

    dataset = blending_datasets(
        args.dataset,
        args.dataset_probs,
        strategy,
        args.seed,
        return_eval=False,
    )
    dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    dataset = SFTDataset(dataset, tokenizer, args.max_len, strategy, pretrain_mode=False)
    dataloader = strategy.setup_dataloader(dataset, args.micro_batch_size, True, False, drop_last=False)
    pbar = tqdm(
        dataloader,
        disable=not strategy.is_rank_0(),
    )

    output_dataset = []
    for _, input_ids, attention_masks, info in pbar:
        inputs = inputs.squeeze(1).to(torch.cuda.current_device())
        attention_mask = attention_masks.squeeze(1).to(torch.cuda.current_device())
        rewards = model(input_ids, attention_masks)
        for prompt, output, reward in zip(info["input"], info["output"], rewards):
            output_dataset.append({"input": prompt, "output": output, "reward": reward.item()})

        with jsonlines.open(args.output_path + str(strategy.get_rank()), mode="w") as writer:
            writer.write_all(output_dataset)

    # wait unitl all processes generate done
    dist.barrier()

    # concate multiple output files in rank 0
    if strategy.is_rank_0():
        output_dataset = []
        world_size = dist.get_world_size()
        files = [args.output_path + str(rank) for rank in range(world_size)]
        for file in files:
            with jsonlines.open(file, mode="r") as reader:
                for obj in reader:
                    output_dataset.append(obj)
            os.remove(file)

        if args.post_processor == "dt":
            strategy.print("Use Decision Transformer")
            decesion_transformer_processor(args, output_dataset)

        with jsonlines.open(args.output_path, mode="w") as writer:
            writer.write_all(output_dataset)


def reward_normalization(objs):
    rewards = [float(obj["reward"]) for obj in objs]
    rewards = torch.tensor(rewards, dtype=torch.float64)
    rewards = (rewards - rewards.mean()) / rewards.std()
    for i, obj in enumerate(objs):
        obj["reward"] = rewards[i].item()


DEFAULT_REWARD_PROMPT = "{input} <rm_score>: {reward} "


def decesion_transformer_processor(args, objs):
    reward_prompt = args.get("reward_template", DEFAULT_REWARD_PROMPT)
    assert "{input}" in reward_prompt
    assert "{reward}" in reward_prompt

    if args.normalize_reward:
        reward_normalization(objs)

    for obj in tqdm(objs, desc="Decision Transformer process..."):
        input = obj["input"]
        reward = "{:.2f}".format(float(obj["reward"]))
        input = reward_prompt.replace("{reward}", reward).replace("{input}", input)
        obj["input"] = input

    return objs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_task", type=str, default=None)
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--load_model", type=str, default=None)
    parser.add_argument("--prompt_max_len", type=int, default=1024)
    parser.add_argument("--max_len", type=int, default=2048)
    parser.add_argument("--zero_stage", type=int, default=0)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--inference_tp_size", type=int, default=1)
    parser.add_argument("--ta_prompt", type=str, default=None)
    parser.add_argument("--greedy_sampling", action="store_true", default=False)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--flash_attn", action="store_true", default=False)

    # batch inference
    parser.add_argument("--micro_batch_size", type=int, default=16)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--dataset_probs", type=str, default="1.0")
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=500000)

    # decision transformer
    parser.add_argument("--post_processor", type=str, default=None)
    parser.add_argument("--normalize_reward", action="store_true", default=False)

    args = parser.parse_args()
    if args.eval_task and args.eval_task == "generate":
        batch_generate(args)
    elif args.eval_task and args.eval_task == "rm":
        batch_rm_inference(args)
    else:
        print("Invalid or missing '--eval_task' argument. Please specify either 'generate' or 'rm'.")
