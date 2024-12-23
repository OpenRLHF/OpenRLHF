import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

def apply_lora(model_name_or_path, output_path, lora_path):
    print(f"Loading the base model from {model_name_or_path}")
    base = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True
    )
    base_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    print(f"Loading the LoRA adapter from {lora_path}")

    lora_model = PeftModel.from_pretrained(
        base,
        lora_path,
        torch_dtype=torch.float16,
    )

    print("Applying the LoRA")
    model = lora_model.merge_and_unload()

    print(f"Saving the target model to {output_path}")
    model.save_pretrained(output_path)
    base_tokenizer.save_pretrained(output_path)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply LoRA to a base model and save the combined model.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the base model directory."
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        required=True,
        help="Path to the LoRA adapter directory."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the combined model."
    )
    args = parser.parse_args()

    apply_lora(args.model_path, args.output_path, args.lora_path)