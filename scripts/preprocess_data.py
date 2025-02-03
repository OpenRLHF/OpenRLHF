import json
import argparse

def add_r1_zero(input_path):
    output_path = input_path.replace(".jsonl", "") + "_add_r1_zero.jsonl"
    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        for line in fin:
            inst = json.loads(line.strip())
            prompt = f"A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\n" + \
                     f"User: {inst['question']} Show your work in great details. And return the final answer in <answer> </answer> tags, for example <answer> the answer is x </answer>. " + \
                     f"Assistant: Let me solve this step by step."
            inst["input_r1_zero"] = prompt
            fout.write(json.dumps(inst, ensure_ascii=False) + "\n")

def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='data')
    group.add_argument('--task-type',
                       type=str,
                       required=True,
                       choices=['add_r1_zero'],
                       help='What type of task to use.')
    group.add_argument('--input-path',
                       type=str,
                       required=True,
                       help='What type of task to use.')
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    func = globals()[args.task_type]
    func(args.input_path)

if __name__ == '__main__':
    main()
