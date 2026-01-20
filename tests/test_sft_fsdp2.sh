#!/bin/bash
set -x

# FSDP2 SFT Training Test Script
# This script runs SFT training with FSDP2 backend for comparison with DeepSpeed

# Configuration
NUM_GPUS=${NUM_GPUS:-8}
MAX_SAMPLES=${MAX_SAMPLES:-1000}
MODEL=${MODEL:-"meta-llama/Meta-Llama-3-8B"}
DATASET=${DATASET:-"Open-Orca/OpenOrca"}
OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoint/test_fsdp2"}
LOG_DIR=${LOG_DIR:-"./logs"}

mkdir -p "${LOG_DIR}"

echo "======================================"
echo "FSDP2 SFT Training Test"
echo "======================================"
echo "GPUs: ${NUM_GPUS}"
echo "Model: ${MODEL}"
echo "Dataset: ${DATASET}"
echo "Max samples: ${MAX_SAMPLES}"
echo "======================================"

read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
   --backend fsdp2 \
   --max_len 512 \
   --dataset ${DATASET} \
   --input_key question \
   --output_key response \
   --train_batch_size 128 \
   --micro_train_batch_size 2 \
   --max_samples ${MAX_SAMPLES} \
   --pretrain ${MODEL} \
   --save_path ${OUTPUT_DIR} \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --max_epochs 1 \
   --param_dtype bf16 \
   --attn_implementation flash_attention_2 \
   --learning_rate 5e-6 \
   --seed 42 \
   --gradient_checkpointing
EOF

# Run with torchrun instead of deepspeed launcher
if [[ ${1} != "slurm" ]]; then
    torchrun --nproc_per_node=${NUM_GPUS} -m $training_commands 2>&1 | tee "${LOG_DIR}/fsdp2_test.log"
fi

echo "FSDP2 test completed. Log saved to ${LOG_DIR}/fsdp2_test.log"
