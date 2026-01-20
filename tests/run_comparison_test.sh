#!/bin/bash
# Full comparison test between DeepSpeed and FSDP2 backends
# Run this inside the Docker container after: pip install -e /openrlhf

set -x

# Configuration
NUM_GPUS=$(nvidia-smi -L | wc -l)
MODEL="meta-llama/Meta-Llama-3-8B"
DATASET="Open-Orca/OpenOrca"
MAX_SAMPLES=200
TRAIN_BATCH_SIZE=64
MICRO_BATCH_SIZE=2
MAX_LEN=256
MAX_EPOCHS=1
LR=5e-6
SEED=42

OUTPUT_DIR="/tmp/comparison_results"
mkdir -p $OUTPUT_DIR

echo "=============================================="
echo "Backend Comparison Test"
echo "=============================================="
echo "GPUs: $NUM_GPUS"
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Max samples: $MAX_SAMPLES"
echo "Train batch size: $TRAIN_BATCH_SIZE"
echo "Micro batch size: $MICRO_BATCH_SIZE"
echo "=============================================="

# Run DeepSpeed test
echo ""
echo "=========================================="
echo "Running DeepSpeed backend..."
echo "=========================================="

deepspeed --num_gpus=$NUM_GPUS --module openrlhf.cli.train_sft \
    --backend deepspeed \
    --max_len $MAX_LEN \
    --dataset $DATASET \
    --input_key question \
    --output_key response \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --micro_train_batch_size $MICRO_BATCH_SIZE \
    --max_samples $MAX_SAMPLES \
    --pretrain $MODEL \
    --save_path "$OUTPUT_DIR/deepspeed_ckpt" \
    --save_steps -1 \
    --logging_steps 1 \
    --eval_steps -1 \
    --zero_stage 2 \
    --max_epochs $MAX_EPOCHS \
    --param_dtype bf16 \
    --attn_implementation flash_attention_2 \
    --learning_rate $LR \
    --seed $SEED \
    --gradient_checkpointing \
    2>&1 | tee "$OUTPUT_DIR/deepspeed.log"

# Run FSDP2 test
echo ""
echo "=========================================="
echo "Running FSDP2 backend..."
echo "=========================================="

torchrun --nproc_per_node=$NUM_GPUS -m openrlhf.cli.train_sft \
    --backend fsdp2 \
    --max_len $MAX_LEN \
    --dataset $DATASET \
    --input_key question \
    --output_key response \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --micro_train_batch_size $MICRO_BATCH_SIZE \
    --max_samples $MAX_SAMPLES \
    --pretrain $MODEL \
    --save_path "$OUTPUT_DIR/fsdp2_ckpt" \
    --save_steps -1 \
    --logging_steps 1 \
    --eval_steps -1 \
    --max_epochs $MAX_EPOCHS \
    --param_dtype bf16 \
    --attn_implementation flash_attention_2 \
    --learning_rate $LR \
    --seed $SEED \
    --gradient_checkpointing \
    2>&1 | tee "$OUTPUT_DIR/fsdp2.log"

# Extract and compare loss values
echo ""
echo "=========================================="
echo "Extracting and comparing loss values..."
echo "=========================================="

python3 << 'EOF'
import re
import sys

def extract_losses(log_file, pattern=r"gpt_loss['\"]?\s*[:=]\s*([0-9]+\.[0-9]+)"):
    """Extract loss values from log file."""
    losses = []
    try:
        with open(log_file, 'r') as f:
            for line in f:
                matches = re.findall(pattern, line)
                for match in matches:
                    try:
                        losses.append(float(match))
                    except ValueError:
                        pass
    except FileNotFoundError:
        print(f"Warning: {log_file} not found")
    return losses

def main():
    ds_losses = extract_losses('/tmp/comparison_results/deepspeed.log')
    fsdp_losses = extract_losses('/tmp/comparison_results/fsdp2.log')
    
    print(f"DeepSpeed losses ({len(ds_losses)} values): {ds_losses[:10]}...")
    print(f"FSDP2 losses ({len(fsdp_losses)} values): {fsdp_losses[:10]}...")
    
    if not ds_losses or not fsdp_losses:
        print("\nWarning: Could not extract enough loss values for comparison")
        return
    
    # Compare first few losses
    n = min(len(ds_losses), len(fsdp_losses), 10)
    
    print(f"\nComparing first {n} loss values:")
    print("-" * 60)
    print(f"{'Step':<6} {'DeepSpeed':<15} {'FSDP2':<15} {'Diff %':<10}")
    print("-" * 60)
    
    max_diff = 0
    for i in range(n):
        ds = ds_losses[i]
        fsdp = fsdp_losses[i]
        diff_pct = abs(ds - fsdp) / max(ds, 1e-10) * 100
        max_diff = max(max_diff, diff_pct)
        status = "✓" if diff_pct < 5 else "✗"
        print(f"{i+1:<6} {ds:<15.6f} {fsdp:<15.6f} {diff_pct:>6.2f}% {status}")
    
    print("-" * 60)
    print(f"Maximum difference: {max_diff:.2f}%")
    
    if max_diff < 5:
        print("\n✓ SUCCESS: Loss values match within 5% tolerance!")
    else:
        print("\n✗ FAILURE: Loss values differ by more than 5%")

if __name__ == "__main__":
    main()
EOF

echo ""
echo "=========================================="
echo "Comparison complete!"
echo "Check $OUTPUT_DIR for detailed logs."
echo "=========================================="
