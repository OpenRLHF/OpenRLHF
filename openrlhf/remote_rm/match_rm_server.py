import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import jsonlines
import re
from openrlhf.remote_rm.grader import grade_answer

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

answer_dict = {item["question"]: {"ref": item["answer"], "type": item["type"]} for item in dataset}

def get_reward(sequences):
    """
    reward: 1 if the answer is correct, -1 if the answer is incorrect, -100 if the answer is invalid
    """
    rewards = []
    for sequence in sequences:
        try:
            q, a = sequence.split("Answer:")
            q = re.sub(r"Question:", "", q).strip()
            a = a.strip()
            v = answer_dict.get(q, None)
            if v is None:
                rewards.append(-1)
                continue
            ref, data_type = v["ref"], v["type"]
            hyp = extract_answer(a, data_type)
            if hyp == "[INVALID]":
                rewards.append(-1)
                print((hyp, ref, rewards[-1]))
                continue
            if grade_answer(hyp, ref):
                rewards.append(1)
            else:
                rewards.append(-1)
            print((hyp, ref, rewards[-1]))
        except Exception as e:
            print(e)
            rewards.append(-1)
    return rewards

app = FastAPI()

class InputText(BaseModel):
    query: List[str]

class OutputPrediction(BaseModel):
    rewards: List[float]

@app.post("/predict", response_model=OutputPrediction)
async def predict(input_text: InputText):
    return {"rewards": get_reward(input_text.query)}