import argparse

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from openrlhf.utils.utils import convert_to_torch_dtype


def apply_lora(model_name_or_path, lora_path, output_path, is_rm, param_dtype):
    print(f"Loading the base model from {model_name_or_path}")
    model_cls = AutoModelForCausalLM if not is_rm else AutoModelForSequenceClassification
    torch_dtype = convert_to_torch_dtype(param_dtype)
    base = model_cls.from_pretrained(model_name_or_path, torch_dtype=torch_dtype, low_cpu_mem_usage=True)
    base_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    print(f"Loading the LoRA adapter from {lora_path}")
    # apply lora to transformer
    lora_model = PeftModel.from_pretrained(
        base,
        lora_path,
        torch_dtype=torch_dtype,  # default: bf16
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
    parser.add_argument(
        "--param_dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16"],
        help="Model data type: 'bf16' uses bfloat16, 'fp16' uses float16",
    )
    args = parser.parse_args()
    apply_lora(args.model_path, args.lora_path, args.output_path, args.is_rm, args.param_dtype)
