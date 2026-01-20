#!/bin/bash
set -x

# FSDP2 + AutoTP + Ring Attention SFT Training Test Script
# This script tests the combined configuration of:
# - FSDP2 for data parallel sharding
# - AutoTP (Tensor Parallelism) using HF's built-in ._tp_plan
# - Ring Attention for sequence parallelism

# Configuration:
# - 8 GPUs total
# - tensor_parallel_size=2, ring_attn_size=2
# - This means DP=2 (2 data parallel groups)
# - Each DP group has 2 TP ranks AND 2 Ring Attention ranks
# - Total: 2 * 2 * 2 = 8 GPUs

NUM_GPUS=${NUM_GPUS:-8}
TP_SIZE=${TP_SIZE:-2}
RING_SIZE=${RING_SIZE:-2}
MAX_SAMPLES=${MAX_SAMPLES:-100}
MODEL=${MODEL:-"meta-llama/Meta-Llama-3-8B"}
DATASET=${DATASET:-"Open-Orca/OpenOrca"}
OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoint/test_fsdp2_tp${TP_SIZE}_ring${RING_SIZE}"}
LOG_DIR=${LOG_DIR:-"./logs"}
MAX_LEN=${MAX_LEN:-2048}  # Longer sequences for Ring Attention

mkdir -p "${LOG_DIR}"

# Calculate DP size
DP_SIZE=$((NUM_GPUS / TP_SIZE / RING_SIZE))

# Validate configuration
if [[ $((DP_SIZE * TP_SIZE * RING_SIZE)) -ne ${NUM_GPUS} ]]; then
    echo "ERROR: NUM_GPUS (${NUM_GPUS}) must equal DP_SIZE * TP_SIZE * RING_SIZE"
    echo "Current: ${DP_SIZE} * ${TP_SIZE} * ${RING_SIZE} = $((DP_SIZE * TP_SIZE * RING_SIZE))"
    exit 1
fi

if [[ ${DP_SIZE} -lt 1 ]]; then
    echo "ERROR: DP_SIZE must be >= 1, got ${DP_SIZE}"
    echo "Reduce TP_SIZE or RING_SIZE"
    exit 1
fi

echo "======================================"
echo "FSDP2 + AutoTP + Ring Attention Test"
echo "======================================"
echo "Total GPUs: ${NUM_GPUS}"
echo "Data Parallel Size: ${DP_SIZE}"
echo "Tensor Parallel Size: ${TP_SIZE}"
echo "Ring Attention Size: ${RING_SIZE}"
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
   --fsdp_tensor_parallel_size ${TP_SIZE} \
   --use_hf_tp_plan \
   --ring_attn_size ${RING_SIZE} \
   --packing_samples
EOF

# Run with torchrun
if [[ ${1} != "slurm" ]]; then
    torchrun --nproc_per_node=${NUM_GPUS} -m $training_commands 2>&1 | tee "${LOG_DIR}/fsdp2_tp${TP_SIZE}_ring${RING_SIZE}_test.log"
fi

echo "FSDP2 + AutoTP + Ring Attention test completed."
echo "Log saved to ${LOG_DIR}/fsdp2_tp${TP_SIZE}_ring${RING_SIZE}_test.log"
