import argparse
import torch
from openrlhf.models import Actor
from openrlhf.utils import get_tokenizer


def generate(args):
    # dummy strategy
    class Empty:
        pass

    dummy_strategy = Empty()
    dummy_strategy.print = print
    dummy_strategy.is_rank_0 = lambda: True
    dummy_strategy.args = args

    # configure model
    model = Actor(
        args.pretrain,
        attn_implementation=args.attn_implementation,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        device_map="auto",
    )

    # configure tokenizer
    tokenizer = get_tokenizer(
        args.pretrain, model.model, "left", dummy_strategy, use_fast=not args.disable_fast_tokenizer
    )

    if args.ta_prompt:
        with open(args.ta_prompt, "r") as f:
            user_prompt = f.read()
    else:
        user_prompt = ""

    if args.apply_chat_template:
        conversations = []

    while True:
        inputs = input("Please enter a prompt (or type 'exit' to quit): ")
        if inputs.strip().lower() == "exit":
            print("Exiting program...")
            break
        if inputs.strip().lower() == "clear":
            if args.apply_chat_template:
                conversations = []
            else:
                user_prompt = ""
            continue

        # get input prompt
        if args.apply_chat_template:
            conversations.append({"role": "user", "content": inputs})
            user_prompt = tokenizer.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)
        else:
            user_prompt = user_prompt + "\n" + args.input_template.format(inputs)

        if args.enable_csft:
            user_prompt += args.csft_prompt.strip() + " "

        user_prompt_len = len(user_prompt)
        input_ids = tokenizer.encode(user_prompt, return_tensors="pt").to(torch.cuda.current_device())
        outputs = model.generate(
            input_ids=input_ids,
            use_cache=True,
            max_length=args.max_len,
            do_sample=not args.greedy_sampling,
            top_p=args.top_p,
            early_stopping=False,
            num_beams=1,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        if args.apply_chat_template:
            generated_ids = outputs[0][:, input_ids.shape[1] :]
            response = tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )[0]
            conversations.append({"role": "assistant", "content": response})
        else:
            user_prompt = tokenizer.batch_decode(outputs[0], skip_special_tokens=True)[0]
            response = user_prompt[user_prompt_len:]

        print(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        help="Attention implementation (e.g., eager, flash_attention_2, flash_attention_3, kernels-community/vllm-flash-attn3)",
    )
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)

    # QLora
    parser.add_argument("--load_in_4bit", action="store_true", default=False, help="Use QLoRA")

    # Sampling
    parser.add_argument("--pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--max_len", type=int, default=4096)
    parser.add_argument("--greedy_sampling", action="store_true", default=False, help="Use Greedy sampling")
    parser.add_argument("--top_p", type=float, default=0.9, help="top_p for Sampling")
    parser.add_argument("--temperature", type=float, default=0.2, help="temperature for Sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--input_template", type=str, default="User: {}\nAssistant: ")
    parser.add_argument(
        "--apply_chat_template", action="store_true", default=False, help="Use HF tokenizer chat template"
    )

    parser.add_argument("--ta_prompt", type=str, default=None)
    parser.add_argument("--enable_csft", action="store_true", default=False)
    parser.add_argument("--csft_prompt", type=str, default="<rm_score>: 5.00", help="conditional SFT prompt")

    # ModelScope parameters
    parser.add_argument("--use_ms", action="store_true", default=False)

    args = parser.parse_args()

    if args.input_template and "{}" not in args.input_template:
        print("[Warning] {} not in args.input_template, set to None")
        args.input_template = None

    if args.input_template and "\\n" in args.input_template:
        print(
            "[Warning] input_template contains \\n characters instead of newline. "
            "You likely want to pass $'\\n' in Bash or \"`n\" in PowerShell."
        )

    if args.use_ms:
        from modelscope.utils.hf_util import patch_hub

        # Patch hub to download models from modelscope to speed up.
        patch_hub()

    print(args)
    generate(args)
