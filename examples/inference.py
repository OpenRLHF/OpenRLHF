import argparse
from collections import OrderedDict

import torch
from utils import get_strategy, get_tokenizer

from openllama2.models import Actor, RewardModel

# from openllama2.models.llama_flash_attn_monkey_patch import (
#     replace_llama_attn_with_flash_attn,
# )
# replace_llama_attn_with_flash_attn()


def generate(args):
    # configure strategy
    strategy = get_strategy(args)

    # configure model
    from_config = bool(args.load_model)
    model = Actor(args.pretrain, from_config)

    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, model.model, "left", strategy)

    # load Pytorch model
    if args.load_model:
        strategy.print("Load model: ", args.load_model)
        strategy.load_model(model, args.load_model)

    # prepare models
    model = strategy.prepare(model)

    model.eval()
    while True:
        user_prompt = input("Please enter a prompt (or type 'exit' to quit): ")
        if user_prompt.strip().lower() == "exit":
            print("Exiting program...")
            break

        # get input prompt
        user_prompt = "\nHuman: " + user_prompt + "\nAssistant: "

        if args.ta_prompt:
            with open(args.ta_prompt, "r") as f:
                ta_prompt = f.read()
            user_prompt = ta_prompt + user_prompt

        input_ids = tokenizer.encode(user_prompt, return_tensors="pt").to(torch.cuda.current_device())
        outputs = model.generate(
            input_ids=input_ids,
            max_length=args.max_len,
            do_sample=True,
            top_p=0.9,
            early_stopping=True,
            num_beams=1,
            temperature=0.5,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        output = tokenizer.batch_decode(outputs[0], skip_special_tokens=True)
        output = output[0].replace(r"\n", "\n")
        if args.ta_prompt:
            output = output[len(ta_prompt) - 1 :]
        print(output)


def rm(args):
    # configure strategy
    strategy = get_strategy(args)

    # configure model
    # load huggingface model/config
    from_config = bool(args.load_model)
    model = RewardModel(args.pretrain, from_config)

    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, model.model, "left", strategy)

    # load SFT model
    if args.load_model:

        def key_replace_fn(states_dict):
            new_state_dict = OrderedDict()
            for k, v in states_dict.items():
                new_state_dict[k.replace("transformer.", "model.")] = v
            return new_state_dict

        strategy.load_model(model, args.load_model, strict=False, key_replace_fn=key_replace_fn)
        strategy.print("Load model: ", args.load_model)

    # prepare models
    model = strategy.prepare(model)
    model.eval()

    while True:
        user_prompt = input("Please enter a prompt (or type 'exit' to quit): ")
        user_chosen_answer = input(
            "Please enter a chosen_answer (or type 'enter' to skip this test or type 'exit' to quit): "
        )
        user_reject_answer = input(
            "Please enter a reject_answer (or type 'enter' to skip this test or type 'exit' to quit): "
        )

        if (
            user_prompt.strip().lower() == "exit"
            or user_chosen_answer.strip().lower() == "exit"
            or user_reject_answer.strip().lower() == "exit"
        ):
            print("Exiting program...")
            break

        if user_chosen_answer:
            chosen_sequence = user_prompt + user_chosen_answer + " " + tokenizer.eos_token
            chosen_sequence_token = tokenizer(chosen_sequence, return_tensors="pt").to(torch.cuda.current_device())
            print(f"chosen_sequence: {chosen_sequence}")
            chosen_reward = model(chosen_sequence_token["input_ids"], chosen_sequence_token["attention_mask"])
            print(f"chosen_reward: {chosen_reward}")
        if user_reject_answer:
            reject_sequence = user_prompt + user_reject_answer + " " + tokenizer.eos_token
            reject_sequence_token = tokenizer(reject_sequence, return_tensors="pt").to(torch.cuda.current_device())
            print(f"reject_sequence: {reject_sequence}")
            reject_reward = model(reject_sequence_token["input_ids"], reject_sequence_token["attention_mask"])
            print(f"reject_reward: {reject_reward}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_task", type=str, default=None)
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--load_model", type=str, default=None)
    parser.add_argument("--max_len", type=int, default=2048)
    parser.add_argument("--zero_stage", type=int, default=0)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--inference_tp_size", type=int, default=1)
    parser.add_argument("--ta_prompt", type=str, default=None)
    args = parser.parse_args()
    if args.eval_task and args.eval_task == "generate":
        generate(args)
    elif args.eval_task and args.eval_task == "rm":
        rm(args)
    else:
        print("Invalid or missing '--eval_task' argument. Please specify either 'generate' or 'rm'.")
