import os
from datasets import load_dataset, Dataset

def process_lean_proof(proof_text):
    """处理Lean证明文本,提取到:= by前的部分作为prompt"""
    lines = proof_text.split("\n")
    prompt_lines = []
    
    for line in lines:
        if ":= by" in line:
            prompt_lines.append(line.split(":= by")[0])
            break
        else:
            prompt_lines.append(line)
    
    return "\n".join(prompt_lines)

def prepare_lean_dataset():
    """准备Lean数据集"""
    # 直接加载parquet文件
    dataset = load_dataset(
        'parquet', 
        data_files='/app/qi/backup/data/RPROVER/Lean-workbook-proofs/data/train-00000-of-00001.parquet'
    )['train']
    
    # 处理proof提取prompt
    processed_data = {
        "context_messages": [],
        "formal_statement": []
    }
    
    for item in dataset:
        prompt = process_lean_proof(item["full_proof"])
        processed_data["context_messages"].append(prompt)
        processed_data["formal_statement"].append(item["problem_id"])
    
    # 创建新数据集
    processed_dataset = Dataset.from_dict(processed_data)
    return processed_dataset

def main():
    output_dir = "/app/qi/backup/data/RPROVER/lean_proofs_data"
    
    dataset = prepare_lean_dataset()
    
    os.makedirs(output_dir, exist_ok=True)
    dataset.save_to_disk(output_dir)

if __name__ == "__main__":
    main()
