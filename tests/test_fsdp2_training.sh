#!/bin/bash
# Test FSDP2 training with a small dataset
# Run this inside the Docker container after: pip install -e /openrlhf

set -ex

# First, test imports
echo "Testing imports..."
python test_fsdp2_import.py

# Get number of GPUs
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Found $NUM_GPUS GPUs"

# Small test with FSDP2
echo ""
echo "=========================================="
echo "Testing FSDP2 training (small scale)..."
echo "=========================================="

torchrun --nproc_per_node=$NUM_GPUS -m openrlhf.cli.train_sft \
    --backend fsdp2 \
    --max_len 256 \
    --dataset Open-Orca/OpenOrca \
    --input_key question \
    --output_key response \
    --train_batch_size 64 \
    --micro_train_batch_size 2 \
    --max_samples 100 \
    --pretrain meta-llama/Meta-Llama-3-8B \
    --save_path ./checkpoint/fsdp2_test \
    --save_steps -1 \
    --logging_steps 1 \
    --eval_steps -1 \
    --max_epochs 1 \
    --param_dtype bf16 \
    --attn_implementation flash_attention_2 \
    --learning_rate 5e-6 \
    --gradient_checkpointing \
    2>&1 | tee fsdp2_test.log

echo ""
echo "=========================================="
echo "FSDP2 test completed! Check fsdp2_test.log for results"
echo "=========================================="
