import argparse

import torch
from utils import get_strategy, get_tokenizer

from openllama2.models import Actor

# from openllama2.models.llama_flash_attn_monkey_patch import (
#     replace_llama_attn_with_flash_attn,
# )
# replace_llama_attn_with_flash_attn()

def eval(args):
    # get input prompt
    input = "\nHuman: " + args.input + "\nAssistant: "
    if args.ta_prompt:
        with open(args.ta_prompt, "r") as f:
            ta_prompt = f.read()
        input = ta_prompt + input

    # configure strategy
    strategy = get_strategy(args)

    # configure model
    with strategy.model_init_context():
        from_config = bool(args.load_model)
        model = Actor(args.pretrain, from_config)

        # load Pytorch model
        if args.load_model:
            strategy.print("Load model: ", args.load_model)
            strategy.load_model(model, args.load_model)

    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, model.model, 'left', strategy)

    # prepare models
    model = strategy.prepare(model)

    model.eval()
    input_ids = tokenizer.encode(input, return_tensors='pt').to(torch.cuda.current_device())
    outputs = model.generate(input_ids=input_ids,
                             max_length=args.max_len,
                             do_sample=True,
                             top_p=0.9,
                             early_stopping=True,
                             num_beams=1,
                             temperature=0.5,
                             repetition_penalty=1.2,
                             pad_token_id=tokenizer.pad_token_id,
                             eos_token_id=tokenizer.eos_token_id)
    output = tokenizer.batch_decode(outputs[0], skip_special_tokens=True)
    output = output[0].replace(r'\n', '\n')
    if args.ta_prompt:
        output = output[len(ta_prompt)-1:]
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy',
                    choices=['naive', 'ddp', 'deepspeed'],
                    default='naive')
    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--load_model', type=str, default=None)
    parser.add_argument('--input', type=str, default='Question: How are you ? Answer:')
    parser.add_argument('--max_len', type=int, default=1024)
    parser.add_argument('--zero_stage', type=int, default=0)
    parser.add_argument('--local_rank', type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument('--bf16', action='store_true', default=False)
    parser.add_argument('--inference_tp_size', type=int, default=1)
    parser.add_argument('--ta_prompt', type=str, default=None)
    args = parser.parse_args()
    eval(args)
