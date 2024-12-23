set -x

python -m openrlhf.utils.lora_combiner \
    --model_path meta-llama/Meta-Llama-3-8B \
    --lora_path ./checkpoint/llama-3-8b-sft \
    --output_path ./checkpoint/llama-3-8b-sft-combined