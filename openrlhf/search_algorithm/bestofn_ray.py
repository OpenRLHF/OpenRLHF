# TODO: 脚本还未经过测试

import re
import torch
import random
import jsonlines
from openrlhf.remote_rm.grader import grade_answer

N = 8
TEMPERATURE = 1
MAX_LENGTH = 1024
strategy = "best"

# 提取答案的方式和任务类型有关，所以这部分代码需要保留
data_fpath_list = [
    "/apdcephfs_sh2/share_300000800/user/antewang/Qwen2.5-Math/evaluation/data/gsm8k/train.jsonl",
    "/apdcephfs_sh2/share_300000800/user/antewang/Qwen2.5-Math/evaluation/data/math/train.jsonl",
    "/apdcephfs_sh2/share_300000800/user/antewang/Qwen2.5-Math/evaluation/data/gsm8k/test.jsonl",
    "/apdcephfs_sh2/share_300000800/user/antewang/Qwen2.5-Math/evaluation/data/math/test.jsonl"
]

dataset = []
for data_fpath in data_fpath_list:
    with jsonlines.open(data_fpath, 'r') as reader:
        for item in reader:
            if "gsm8k" in data_fpath:
                question = item["question"]
                answer = item["answer"].split("####")[-1].strip()
                dataset.append({"question": question, "answer": answer, "type": "gsm8k"})
            else:
                question = item["problem"]
                answer = item["answer"]
                dataset.append({"question": question, "answer": answer, "type": "math"})

answer_dict = {item["question"].strip(): {"ref": item["answer"], "type": item["type"]} for item in dataset}

def clean_pad_token(text, pad_token):
    return re.sub(pad_token, "", text)

def clean_eos(text, eos_token):
    return text[:-len(eos_token)] if text.endswith(eos_token) else text

def extract_numbers(text):
    ANS_RE = re.compile(r"(\-?[0-9\.\,]+)")
    match = ANS_RE.search(text)
    try:
        return match.group(1).strip()
    except:
        return "[INVALID]"

def extract_answer(text, data_type):
    if "he answer is:" in text:
        text_split = text.split("he answer is:")
    elif "he answer is" in text:
        text_split = text.split("he answer is")
    else:
        return "[INVALID]"
    
    if len(text_split) == 2:
        content = text_split[-1].strip()
        if len(content) == 0:
            return "[INVALID]"
        if content[-1] == ".":
            content = content[:-1]
        if data_type == "gsm8k":
            content = extract_numbers(content)
        return content
    return "[INVALID]"

def get_full_trajs(query, tokenizer, llms):
    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        top_p=1,
        top_k=-1,
        max_tokens=1024,
        min_tokens=1,
        skip_special_tokens=False,
        include_stop_str_in_output=True,
        logprobs = 0,
    )

    # Expand prompt list based on the number of samples per prompt
    all_prompts = [query] * N
    all_prompt_token_ids = tokenizer(all_prompts, add_special_tokens=False, max_length=MAX_LENGTH, truncation=True,)["input_ids"]

    # Distribute requests to engines and collect responses to outputs
    all_output_refs = []
    batch_size = (len(all_prompt_token_ids) + len(llms) - 1) // len(llms)
    for i, llm in enumerate(llms):
        prompt_token_ids = all_prompt_token_ids[i * batch_size : (i + 1) * batch_size]
        if prompt_token_ids:
            all_output_refs.append(
                llm.generate.remote(sampling_params=sampling_params, prompt_token_ids=prompt_token_ids)
            )

    # Retrieve and combine results from all outputs
    all_outputs = sum(ray.get(all_output_refs), [])
    sequences = [output.outputs[0].text for output in all_outputs]
    cumulative_logprobs = [output.outputs[0].cumulative_logprob for output in all_outputs]
    # custom logprobs, 目前这样写和上面cumulative_logprob一样
    custom_logprobs = [sum(list(item.values())[0].logprob for item in output.outputs[0].logprobs) for output in all_outputs]
    return sequences, cumulative_logprob, custom_logprobs

# 在该脚本中使用PPO中的critic，代码会卡住，别用
def search(query, tokenizer, actor, critic=None):
    trajs, cumulative_logprobs, custom_logprobs = get_full_trajs(query, tokenizer, actor)
    
    if strategy == "best":
        best_idx = torch.argmax(cumulative_logprob)
        return [trajs[best_idx]]
    elif strategy == "voting":
        question = query[len("Question:"): -len("Answer: Let's think step by step\n")].strip()
        predictions = {}
        ref, data_type = v["ref"], v["type"]
        has_gold = False
        for traj, score in zip(trajs, cumulative_logprobs.tolist()):
            hyp = extract_answer(clean_eos(traj, tokenizer.eos_token), data_type)
            if hyp == "[INVALID]":
                continue
            flag = False
            for k in predictions:
                if grade_answer(k, hyp): # simple string match is also ok, this method would be better
                    flag = True
                    predictions[k]["score"] += score
                    predictions[k]["trajs"].append(traj)
                    break
            if not flag:
                predictions[hyp] = {"score": score, "trajs": [traj]}
        sorted_trajs = sorted(predictions.values(), key=lambda x: x["score"] / len(x["trajs"]), reverse=True)
        return [random.choice(sorted_trajs[0]["trajs"])]
