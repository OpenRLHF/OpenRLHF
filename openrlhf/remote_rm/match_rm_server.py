import traceback
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from collections import defaultdict
import jsonlines
import re
from openrlhf.remote_rm.grader import grade_answer
from openrlhf.remote_rm.qwen_math_eval_toolkit.grader import math_equal as qwen_math_equal
from openrlhf.remote_rm.qwen_math_eval_toolkit.parser import extract_answer as qwen_extract_answer
from multiprocessing import Process, Queue

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

answer_dict = {item["question"].strip(): {"ref": item["answer"], "type": item["type"]} for item in dataset}

def get_reward_r1_zero(sequences):
    rewards = []
    for sequence in sequences:
        try:
            q = sequence.split("\nUser: ", 1)[1]
            q = q.split(" Show your work in great details.", 1)[0].strip()
            answer_pattern = r'<answer>(.*?)</answer>'
            match = re.search(answer_pattern, sequence)
            if not match:
                rewards.append(-1.0)
                continue
            a = match.group(1).strip()
            if a == "":
                print(f"!!! Empty answer: {sequence}")
                rewards.append(-1.0)
                continue
            v = answer_dict.get(q, None)
            if v is None:
                print(f"!!! Unmatched question: {q}")
                rewards.append(-1.0)
                continue
            ref, data_type = v["ref"], v["type"]
            hyp = extract_answer(a, data_type)
            if hyp == "[INVALID]":
                print(f"!!! Fail to extract answer from {a} for data_type {data_type}")
                rewards.append(-1.0)
                continue
            if grade_answer(hyp, ref):
                rewards.append(1.0)
            else:
                rewards.append(-0.1)
            print((a, ref, rewards[-1]))
        except Exception as e:
            print(e)
            rewards.append(-1.0)
    return rewards


def qwen_math_equal_subprocess(prediction, reference,  timeout_seconds=10):
    def worker(q, prediction, reference):
        result = qwen_math_equal(prediction=prediction, reference=reference, timeout=False)
        q.put(result)

    q = Queue()
    p = Process(target=worker, args=(q, prediction, reference))
    p.start()

    # 添加超时处理
    p.join(timeout=timeout_seconds)  # 等待进程完成，最多等待 timeout_seconds 秒

    if p.is_alive():
        p.terminate()  # 发送 SIGTERM
        p.join(timeout=1)  # 等待1秒
        if p.is_alive():
            print("force kill")
            p.kill()  # 发送 SIGKILL
            p.join()
        return False

    # 如果进程正常完成，获取结果
    try:
        return q.get_nowait()
    except:
        return False


def get_reward_qwen_math(sequences):
    rewards = []
    for sequence in sequences:
        try:
            query, model_output = sequence.split("<|im_end|>\n<|im_start|>assistant")
            question = query.split("<|im_start|>user")[1].strip()
            model_output = model_output.strip()
            if question not in answer_dict:
                print(f"!!! Unmatched question: {question}")
                rewards.append(-1.0)
                continue

            stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
            for stop_word in stop_words:
                if stop_word in model_output:
                    model_output = model_output.split(stop_word)[0].strip()

            if "boxed" not in model_output:
                print(f"!!! No 'boxed' found: {model_output}")
                box_match = -1.0
            else:
                extract_answer = qwen_extract_answer(model_output, data_name="math")
                answer = answer_dict[question]["ref"]
                if qwen_math_equal_subprocess(prediction=extract_answer, reference=answer):
                    box_match = 1.0
                else:
                    box_match = -0.5
            rewards.append(box_match)
        except Exception as e:
            print(f"!!!Exception: {e}")
            rewards.append(-1.0)
    rewards_dict = defaultdict(int)
    for r in rewards:
        rewards_dict[r] += 1
    print(f"!!! Reward Mean: {sum(rewards) / (len(rewards) + 1e-5)}, Distribution: {rewards_dict}")
    return rewards


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

@app.post("/predict_r1_zero", response_model=OutputPrediction)
async def predict(input_text: InputText):
    return {"rewards": get_reward_r1_zero(input_text.query)}

@app.post("/predict_qwen_math", response_model=OutputPrediction)
async def predict(input_text: InputText):
    return {"rewards": get_reward_qwen_math(input_text.query)}
