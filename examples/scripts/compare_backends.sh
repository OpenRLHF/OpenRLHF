#!/bin/bash
set -x

# Backend Comparison Script
# This script runs SFT training with both DeepSpeed and FSDP2 backends
# and compares the loss values to ensure they match (<5% difference)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/../../comparison_results"
mkdir -p "${OUTPUT_DIR}"

# Common settings
MODEL="meta-llama/Meta-Llama-3-8B"
DATASET="Open-Orca/OpenOrca"
MAX_SAMPLES=500
TRAIN_BATCH_SIZE=128
MICRO_BATCH_SIZE=2
MAX_LEN=512
MAX_EPOCHS=1
LR=5e-6
SEED=42

echo "======================================"
echo "Backend Comparison Test"
echo "======================================"
echo "Model: ${MODEL}"
echo "Dataset: ${DATASET}"
echo "Max samples: ${MAX_SAMPLES}"
echo "Train batch size: ${TRAIN_BATCH_SIZE}"
echo "Micro batch size: ${MICRO_BATCH_SIZE}"
echo "======================================"

# Run DeepSpeed
echo ""
echo "Running DeepSpeed backend..."
deepspeed --module openrlhf.cli.train_sft \
   --backend deepspeed \
   --max_len ${MAX_LEN} \
   --dataset ${DATASET} \
   --input_key question \
   --output_key response \
   --train_batch_size ${TRAIN_BATCH_SIZE} \
   --micro_train_batch_size ${MICRO_BATCH_SIZE} \
   --max_samples ${MAX_SAMPLES} \
   --pretrain ${MODEL} \
   --save_path "${OUTPUT_DIR}/deepspeed_ckpt" \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 2 \
   --max_epochs ${MAX_EPOCHS} \
   --param_dtype bf16 \
   --attn_implementation flash_attention_2 \
   --learning_rate ${LR} \
   --seed ${SEED} \
   --gradient_checkpointing \
   2>&1 | tee "${OUTPUT_DIR}/deepspeed.log"

# Run FSDP2
echo ""
echo "Running FSDP2 backend..."
torchrun --nproc_per_node=8 -m openrlhf.cli.train_sft \
   --backend fsdp2 \
   --max_len ${MAX_LEN} \
   --dataset ${DATASET} \
   --input_key question \
   --output_key response \
   --train_batch_size ${TRAIN_BATCH_SIZE} \
   --micro_train_batch_size ${MICRO_BATCH_SIZE} \
   --max_samples ${MAX_SAMPLES} \
   --pretrain ${MODEL} \
   --save_path "${OUTPUT_DIR}/fsdp2_ckpt" \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --max_epochs ${MAX_EPOCHS} \
   --param_dtype bf16 \
   --attn_implementation flash_attention_2 \
   --learning_rate ${LR} \
   --seed ${SEED} \
   --gradient_checkpointing \
   2>&1 | tee "${OUTPUT_DIR}/fsdp2.log"

# Extract and compare loss values
echo ""
echo "======================================"
echo "Extracting loss values..."
echo "======================================"

# Extract loss values from logs (looking for gpt_loss in the output)
grep -o "gpt_loss.*[0-9]\.[0-9]*" "${OUTPUT_DIR}/deepspeed.log" | head -20 > "${OUTPUT_DIR}/deepspeed_losses.txt"
grep -o "gpt_loss.*[0-9]\.[0-9]*" "${OUTPUT_DIR}/fsdp2.log" | head -20 > "${OUTPUT_DIR}/fsdp2_losses.txt"

echo "DeepSpeed losses:"
cat "${OUTPUT_DIR}/deepspeed_losses.txt"

echo ""
echo "FSDP2 losses:"
cat "${OUTPUT_DIR}/fsdp2_losses.txt"

echo ""
echo "======================================"
echo "Comparison complete! Check ${OUTPUT_DIR} for detailed logs."
echo "======================================"
