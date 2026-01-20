#!/bin/bash
set -x

# FSDP2 SFT Training Test Script
# This script runs SFT training with FSDP2 backend for comparison with DeepSpeed

read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
   --backend fsdp2 \
   --max_len 512 \
   --dataset Open-Orca/OpenOrca \
   --input_key question \
   --output_key response \
   --train_batch_size 128 \
   --micro_train_batch_size 2 \
   --max_samples 1000 \
   --pretrain meta-llama/Meta-Llama-3-8B \
   --save_path ./checkpoint/llama3-8b-sft-fsdp2 \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --max_epochs 1 \
   --param_dtype bf16 \
   --attn_implementation flash_attention_2 \
   --learning_rate 5e-6 \
   --gradient_checkpointing
EOF

# Run with torchrun instead of deepspeed launcher
if [[ ${1} != "slurm" ]]; then
    torchrun --nproc_per_node=8 -m $training_commands
fi
