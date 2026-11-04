import argparse

from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from openrlhf.utils.config import hierarchize
from openrlhf.utils.utils import convert_to_torch_dtype


def apply_lora(model_name_or_path, lora_path, output_path, is_rm, param_dtype):
    print(f"Loading the base model from {model_name_or_path}")
    torch_dtype = convert_to_torch_dtype(param_dtype)
    if is_rm:
        from openrlhf.models import get_llm_for_sequence_regression

        # Restore the scalar head and reward metadata used during training.
        base = get_llm_for_sequence_regression(
            model_name_or_path,
            "reward",
            config=AutoConfig.from_pretrained(lora_path, trust_remote_code=True),
            param_dtype=param_dtype,
            attn_implementation="eager",
        )
    else:
        base = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=torch_dtype, low_cpu_mem_usage=True
        )
    base_tokenizer = AutoTokenizer.from_pretrained(lora_path if is_rm else model_name_or_path)

    print(f"Loading the LoRA adapter from {lora_path}")
    # apply lora to transformer
    lora_model = PeftModel.from_pretrained(
        base,
        lora_path,
        torch_dtype=torch_dtype,  # default: bf16
    )

    print("Applying and merging the LoRA weights")
    base = lora_model.merge_and_unload()

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
        help="Whether to merge an OpenRLHF reward model with a scalar value head",
    )
    parser.add_argument(
        "--ds.param_dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16"],
        help="Model data type: 'bf16' uses bfloat16, 'fp16' uses float16",
    )
    args = hierarchize(parser.parse_args())
    apply_lora(args.model_path, args.lora_path, args.output_path, args.is_rm, args.ds.param_dtype)
