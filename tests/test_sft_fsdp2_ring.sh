#!/bin/bash
set -x

# FSDP2 + Ring Attention SFT Training Test Script
# This script runs SFT training with FSDP2 backend and Ring Attention enabled
# Ring Attention allows training on sequences longer than single-GPU memory capacity

# Configuration:
# - 8 GPUs total with ring_attn_size=2
# - ring_attn_size=2 means we have DP=4 (4 data parallel groups)
# - Each DP group has 2 ranks participating in Ring Attention
# - This enables training on longer sequences by distributing them in a ring topology

NUM_GPUS=${NUM_GPUS:-8}
RING_SIZE=${RING_SIZE:-2}
MAX_SAMPLES=${MAX_SAMPLES:-200}
MODEL=${MODEL:-"meta-llama/Meta-Llama-3-8B"}
DATASET=${DATASET:-"Open-Orca/OpenOrca"}
OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoint/test_fsdp2_ring${RING_SIZE}"}
LOG_DIR=${LOG_DIR:-"./logs"}
MAX_LEN=${MAX_LEN:-4096}  # Use longer sequences for Ring Attention testing

mkdir -p "${LOG_DIR}"

# Calculate DP size
DP_SIZE=$((NUM_GPUS / RING_SIZE))

echo "======================================"
echo "FSDP2 + Ring Attention SFT Training Test"
echo "======================================"
echo "Total GPUs: ${NUM_GPUS}"
echo "Ring Attention Size: ${RING_SIZE}"
echo "Data Parallel Size: ${DP_SIZE}"
echo "Max Sequence Length: ${MAX_LEN}"
echo "Model: ${MODEL}"
echo "Dataset: ${DATASET}"
echo "Max samples: ${MAX_SAMPLES}"
echo "======================================"

# Note: Ring Attention requires packing_samples to be enabled
read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
   --backend fsdp2 \
   --max_len ${MAX_LEN} \
   --dataset ${DATASET} \
   --input_key question \
   --output_key response \
   --train_batch_size 64 \
   --micro_train_batch_size 1 \
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
   --ring_attn_size ${RING_SIZE} \
   --packing_samples
EOF

# Run with torchrun
if [[ ${1} != "slurm" ]]; then
    torchrun --nproc_per_node=${NUM_GPUS} -m $training_commands 2>&1 | tee "${LOG_DIR}/fsdp2_ring${RING_SIZE}_test.log"
fi

echo "FSDP2 + Ring Attention test completed. Log saved to ${LOG_DIR}/fsdp2_ring${RING_SIZE}_test.log"
