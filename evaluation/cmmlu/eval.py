# This code is modified from C-Eval Project: https://github.com/SJTU-LIT/ceval
import argparse
import json
import os
import time
from collections import defaultdict
from glob import glob

import pandas as pd
import torch
from categories import categories, name_en2zh, subcategories
from llama2_evaluator import Llama_Evaluator

choices = ["A", "B", "C", "D"]

category2subject = defaultdict(list)
for k, v in categories.items():
    for subject, subcat in subcategories.items():
        for c in subcat:
            if c in v:
                category2subject[k].append(subject)
category2subject_list = defaultdict(list)
for key, value in category2subject.items():
    for val in value:
        category2subject_list[val] = [val, name_en2zh[val], key]
category2subject = category2subject_list
choices = ["A", "B", "C", "D"]


def main(args, evaluator, take):
    subject_mapping = category2subject  # json.load(f)
    filenames = [s.split("/")[-1] for s in glob(args.input_dir + "/test/*csv")]
    subject_list = [val_file.replace(".csv", "") for val_file in filenames]
    accuracy, summary = {}, {}

    run_date = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
    output_dir = args.output_dir
    save_result_dir = os.path.join(output_dir, f"take{take}")
    if not os.path.exists(save_result_dir):
        os.makedirs(save_result_dir, exist_ok=True)

    all_answers = {}
    for index, subject_name in enumerate(subject_list):
        print(
            f"{index/len(subject_list)} Inference starts at {run_date} on {args.pretrain_model_path} with subject of {subject_name}!"
        )
        val_file_path = os.path.join(args.input_dir + "/test", f"{subject_name}.csv")
        dev_file_path = os.path.join(args.input_dir + "/dev", f"{subject_name}.csv")

        val_df = pd.read_csv(val_file_path)
        dev_df = pd.read_csv(dev_file_path) if args.few_shot else None

        correct_ratio, answers = evaluator.eval_subject(
            subject_name,
            val_df,
            dev_df,
            save_result_dir=save_result_dir if args.do_save_csv else None,
            few_shot=args.few_shot,
            cot=args.cot,
            with_prompt=args.with_prompt,
            constrained_decoding=args.constrained_decoding,
            do_test=False,
        )
        print(f"Subject: {subject_name}")
        print(f"Acc: {correct_ratio}")
        accuracy[subject_name] = correct_ratio
        summary[subject_name] = {
            "score": correct_ratio,
            "num": len(val_df),
            "correct": correct_ratio * len(val_df) / 100,
        }
        all_answers[subject_name] = answers

    json.dump(all_answers, open(save_result_dir + "/submission.json", "w"), ensure_ascii=False, indent=4)
    print("\n\nModel:", args.pretrain_model_path)
    print("Accuracy:")
    for k, v in accuracy.items():
        print(k, ": ", v)

    total_num = 0
    total_correct = 0
    summary["grouped"] = {
        "China specific": {"correct": 0.0, "num": 0},
        "STEM": {"correct": 0.0, "num": 0},
        "Social Science": {"correct": 0.0, "num": 0},
        "Humanities": {"correct": 0.0, "num": 0},
        "Other": {"correct": 0.0, "num": 0},
    }
    for subj, info in subject_mapping.items():
        group = info[2]
        summary["grouped"][group]["num"] += summary[subj]["num"]
        summary["grouped"][group]["correct"] += summary[subj]["correct"]
    for group, info in summary["grouped"].items():
        info["score"] = info["correct"] / info["num"]
        total_num += info["num"]
        total_correct += info["correct"]
    summary["All"] = {"score": total_correct / total_num, "num": total_num, "correct": total_correct}

    json.dump(summary, open(save_result_dir + "/summary.json", "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--pretrain_model_path", type=str)
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--cot", choices=["False", "True"], default="False")
    parser.add_argument("--few_shot", choices=["False", "True"], default="True")
    parser.add_argument("--with_prompt", choices=["False", "True"], default="False")
    parser.add_argument("--constrained_decoding", choices=["False", "True"], default="False")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--n_times", default=1, type=int)
    parser.add_argument("--do_save_csv", choices=["False", "True"], default="False")
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--verbose", action="store_true", help="Print detailed information of each example.")

    args = parser.parse_args()

    args.cot = args.cot == "True"
    args.few_shot = args.few_shot == "True"
    args.with_prompt = args.with_prompt == "True"
    args.do_save_csv = args.do_save_csv == "True"
    args.constrained_decoding = args.constrained_decoding == "True"
    if args.constrained_decoding is True:
        args.n_times = max(args.n_times, 1)
    print(args)

    device = torch.device(0)
    print(device)
    evaluator = Llama_Evaluator(
        args=args,
        choices=choices,
        k=args.ntrain,
        pretrain_model_path=args.pretrain_model_path,
        device=device,
        ckpt_path=args.ckpt_path,
        temperature=args.temperature,
        verbose=args.verbose,
    )
    for i in range(args.n_times):
        main(args, evaluator=evaluator, take=i)
