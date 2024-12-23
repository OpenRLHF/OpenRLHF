set -x

python -m openrlhf.cli.lora_combiner \
    --model_path meta-llama/Meta-Llama-3-8B \
    --lora_path ./checkpoint/llama-3-8b-sft \
    --output_path ./checkpoint/llama-3-8b-sft-combined \
    --bf16