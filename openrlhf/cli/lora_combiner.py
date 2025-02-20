import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer


def apply_lora(model_name_or_path, lora_path, output_path, is_rm, bf16):
    print(f"Loading the base model from {model_name_or_path}")
    model_cls = AutoModelForCausalLM if not is_rm else AutoModelForSequenceClassification
    base = model_cls.from_pretrained(
        model_name_or_path, torch_dtype=torch.bfloat16 if bf16 else "auto", low_cpu_mem_usage=True
    )
    base_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    print(f"Loading the LoRA adapter from {lora_path}")
    # apply lora to transformer
    lora_model = PeftModel.from_pretrained(
        base,
        lora_path,
        torch_dtype=torch.bfloat16 if bf16 else "auto",
    )

    print("Applying and merging the LoRA weights")
    lora_model.merge_and_unload()

    print(f"Saving the complete model to {output_path}")
    base.save_pretrained(output_path)
    base_tokenizer.save_pretrained(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply LoRA to a base model and save the combined model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the base model directory.")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to the LoRA adapter directory.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the combined model.")
    parser.add_argument(
        "--is_rm",
        action="store_true",
        default=False,
        help="Whether to treat the model as a reward model (AutoModelForSequenceClassification)",
    )
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    args = parser.parse_args()
    apply_lora(args.model_path, args.lora_path, args.output_path, args.is_rm, args.bf16)
