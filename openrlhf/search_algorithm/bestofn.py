import re
import torch
import random
import jsonlines
from openrlhf.remote_rm.grader import grade_answer

N = 8
TEMPERATURE = 1
strategy = "voting"

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

answer_dict = {item["question"]: {"ref": item["answer"], "type": item["type"]} for item in dataset}

def clean_pad_token(text, pad_token):
    return re.sub(pad_token, "", text)

def get_full_traj(traj, tokenizer, actor):
    input_ids = tokenizer(traj, return_tensors="pt")
    input_ids = {k: v.to(actor.model.device) for k, v in input_ids.items()}
    outputs = actor.model.generate(**input_ids, do_sample=True, max_new_tokens=1024,
                             temperature=TEMPERATURE, tokenizer=tokenizer, num_return_sequences=N,
                             pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    sequences = tokenizer.batch_decode(outputs, keep_special_tokens=True)
    sequences = [clean_pad_token(seq, tokenizer.pad_token) for seq in sequences]
    return sequences

def get_scores(trajs, tokenizer, critic):
    input_ids = tokenizer(trajs, return_tensors="pt", padding=True, truncation=True)
    input_ids = {k: v.to(critic.device) for k, v in input_ids.items()}
    outputs = critic.compute_value(**input_ids, return_dict=False)
    return (torch.clamp(outputs[0].squeeze(), min=-1, max=1) + 1) / 2

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
        if content[-1] == ".":
            content = content[:-1]
        if data_type == "gsm8k":
            content = extract_numbers(content)
        return content
    return "[INVALID]"

def clean_eos(traj, eos_token):
    if traj.endswith(eos_token):
        return traj[:-len(eos_token)]
    else:
        return traj

def search(query, tokenizer, actor, critic):
    trajs = get_full_traj(query, tokenizer, actor)
    scores = get_scores(trajs, tokenizer, critic)
    if strategy == "bestofn":
        best_idx = torch.argmax(scores)
        return [trajs[best_idx]]
    elif strategy == "bestandworst":
        best_idx = torch.argmax(scores)
        worse_idx = torch.argmin(scores)
        return [trajs[best_idx], trajs[worse_idx]]
    elif strategy == "random":
        return random.sample(trajs, 2)
    elif strategy == "voting":
        # question
        question = query[len("Question:"): -len("Answer: Let's think step by step\n")].strip()
        v = answer_dict.get(question, None)
        if v is None:
            return []
        predictions = {}
        ref, data_type = v["ref"], v["type"]
        has_gold = False
        for traj, score in zip(trajs, scores.tolist()):
            hyp = extract_answer(clean_eos(traj, tokenizer.eos_token), data_type)
            if hyp == "[INVALID]":
                continue
            if grade_answer(hyp, ref):
                score += 1
                has_gold = True
            flag = False
            for k in predictions:
                if grade_answer(k, hyp):
                    flag = True
                    predictions[k]["score"] += score
                    predictions[k]["trajs"].append(traj)
                    break
            if not flag:
                predictions[hyp] = {"score": score, "trajs": [traj]}
        sorted_trajs = sorted(predictions.values(), key=lambda x: x["score"], reverse=True)
        if len(sorted_trajs) == 1 or not has_gold:
            return [random.choice(sorted_trajs[0]["trajs"])]
        else:
            return [random.choice(_sorted_trajs["trajs"]) for _sorted_trajs in sorted_trajs[:2]]
