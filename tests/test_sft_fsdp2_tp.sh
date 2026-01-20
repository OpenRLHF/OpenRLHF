#!/bin/bash
set -x

# FSDP2 + AutoTP (Tensor Parallelism) SFT Training Test Script
# This script runs SFT training with FSDP2 backend and tensor parallelism enabled
# Using HuggingFace's built-in ._tp_plan for automatic tensor parallelism

# Configuration:
# - 8 GPUs total
# - tensor_parallel_size=2 means we have DP=4 (4 data parallel groups)
# - Each DP group has 2 TP ranks

NUM_GPUS=${NUM_GPUS:-8}
TP_SIZE=${TP_SIZE:-2}
MAX_SAMPLES=${MAX_SAMPLES:-500}
MODEL=${MODEL:-"meta-llama/Meta-Llama-3-8B"}
DATASET=${DATASET:-"Open-Orca/OpenOrca"}
OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoint/test_fsdp2_tp${TP_SIZE}"}
LOG_DIR=${LOG_DIR:-"./logs"}

mkdir -p "${LOG_DIR}"

# Calculate DP size
DP_SIZE=$((NUM_GPUS / TP_SIZE))

echo "======================================"
echo "FSDP2 + AutoTP SFT Training Test"
echo "======================================"
echo "Total GPUs: ${NUM_GPUS}"
echo "Tensor Parallel Size: ${TP_SIZE}"
echo "Data Parallel Size: ${DP_SIZE}"
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
   --gradient_checkpointing \
   --fsdp_tensor_parallel_size ${TP_SIZE} \
   --use_hf_tp_plan
EOF

# Run with torchrun
if [[ ${1} != "slurm" ]]; then
    torchrun --nproc_per_node=${NUM_GPUS} -m $training_commands 2>&1 | tee "${LOG_DIR}/fsdp2_tp${TP_SIZE}_test.log"
fi

echo "FSDP2 + AutoTP test completed. Log saved to ${LOG_DIR}/fsdp2_tp${TP_SIZE}_test.log"
