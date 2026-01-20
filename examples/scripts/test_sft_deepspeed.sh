#!/bin/bash
set -x

# DeepSpeed SFT Training Test Script
# This script runs SFT training with DeepSpeed backend for comparison with FSDP2

read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
   --backend deepspeed \
   --max_len 512 \
   --dataset Open-Orca/OpenOrca \
   --input_key question \
   --output_key response \
   --train_batch_size 128 \
   --micro_train_batch_size 2 \
   --max_samples 1000 \
   --pretrain meta-llama/Meta-Llama-3-8B \
   --save_path ./checkpoint/llama3-8b-sft-deepspeed \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 2 \
   --max_epochs 1 \
   --param_dtype bf16 \
   --attn_implementation flash_attention_2 \
   --learning_rate 5e-6 \
   --gradient_checkpointing
EOF

# Run with deepspeed launcher
if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi
