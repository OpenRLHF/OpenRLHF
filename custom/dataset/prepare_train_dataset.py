import jsonlines, json
from transformers import AutoTokenizer


gsm8k_path = "/apdcephfs/private_antewang/RLVF/data_qwen/Qwen2.5-Math/evaluation/data/gsm8k/train.jsonl"
math_path = "/apdcephfs/private_antewang/RLVF/data_qwen/Qwen2.5-Math/evaluation/data/math/train.jsonl"

tokenizer = AutoTokenizer.from_pretrained("/apdcephfs_cq11/share_1603164/data/antewang/trained_models/Qwen-2.5-Math-1.5B_sft_star_fp32/checkpoint-1738")

dataset = []
filtered = 0
with jsonlines.open(gsm8k_path) as reader:
    for item in reader:
        question = item["question"]
        if len(tokenizer.tokenize(question)) > 512:
            filtered += 1
            continue
        answer = item["answer"].split("####")[-1].strip()
        dataset.append({"question": question, "answer": answer, "type": "gsm8k"})

with jsonlines.open(math_path) as reader:
    for item in reader:
        question = item["problem"]
        if len(tokenizer.tokenize(question)) > 512:
            filtered += 1
            continue
        answer = item["answer"]
        dataset.append({"question": question, "answer": answer, "type": "math"})

print(filtered, len(dataset))

def wrap_query(question):
    return f"Question: {question}\n\nAnswer: Let's think step by step\n"

for item in dataset:
    item["input"] = wrap_query(item["question"])

print(len(dataset))

out_fpath = f"train_gsm8k_math.jsonl"
with open(out_fpath, "w") as f:
    f.write("\n".join(json.dumps(instance) for instance in dataset))


    
