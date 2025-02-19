
import jsonlines

DEFAULT_N = 8
DEFAULT_TEMPERATURE = 1
DEFAULT_MAX_LENGTH = 1024
strategy = "best"

def initialize_question_answer_map():
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
    return answer_dict
