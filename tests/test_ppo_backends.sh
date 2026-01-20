#!/bin/bash
set -x

# PPO Backend Comparison Script
# Tests PPO training with both DeepSpeed and FSDP2 backends
# and compares reward/KL metrics

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/../ppo_comparison_results"
mkdir -p "${OUTPUT_DIR}"

# Common settings - use small model and limited samples for quick testing
MODEL="Qwen/Qwen2.5-0.5B"
REWARD_MODEL="Qwen/Qwen2.5-0.5B"  # Use same model as reward for testing
MAX_SAMPLES=100
TRAIN_BATCH_SIZE=16
MICRO_BATCH_SIZE=2
ROLLOUT_BATCH_SIZE=32
PROMPT_MAX_LEN=256
GENERATE_MAX_LEN=128
MAX_EPOCHS=1
LR=1e-6
SEED=42
NUM_GPUS=4

echo "======================================"
echo "PPO Backend Comparison Test"
echo "======================================"
echo "Model: ${MODEL}"
echo "Max samples: ${MAX_SAMPLES}"
echo "Train batch size: ${TRAIN_BATCH_SIZE}"
echo "Micro batch size: ${MICRO_BATCH_SIZE}"
echo "Rollout batch size: ${ROLLOUT_BATCH_SIZE}"
echo "Number of GPUs: ${NUM_GPUS}"
echo "======================================"

# Run DeepSpeed PPO
echo ""
echo "Running DeepSpeed PPO backend..."
python3 -m openrlhf.cli.train_ppo_ray \
   --backend deepspeed \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node ${NUM_GPUS} \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node ${NUM_GPUS} \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node ${NUM_GPUS} \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node ${NUM_GPUS} \
   --vllm_num_engines 2 \
   --vllm_tensor_parallel_size 1 \
   --colocate_all_models \
   --vllm_gpu_memory_utilization 0.3 \
   --pretrain ${MODEL} \
   --reward_pretrain ${REWARD_MODEL} \
   --save_path "${OUTPUT_DIR}/deepspeed_ppo" \
   --ckpt_path "${OUTPUT_DIR}/deepspeed_ppo_ckpt" \
   --train_batch_size ${TRAIN_BATCH_SIZE} \
   --rollout_batch_size ${ROLLOUT_BATCH_SIZE} \
   --micro_train_batch_size ${MICRO_BATCH_SIZE} \
   --n_samples_per_prompt 1 \
   --max_epochs ${MAX_EPOCHS} \
   --prompt_max_len ${PROMPT_MAX_LEN} \
   --generate_max_len ${GENERATE_MAX_LEN} \
   --max_samples ${MAX_SAMPLES} \
   --zero_stage 2 \
   --param_dtype bf16 \
   --actor_learning_rate ${LR} \
   --critic_learning_rate ${LR} \
   --init_kl_coef 0.01 \
   --prompt_data Open-Orca/OpenOrca \
   --input_key question \
   --normalize_reward \
   --gradient_checkpointing \
   --vllm_sync_backend nccl \
   --enforce_eager \
   --seed ${SEED} \
   --logging_steps 1 \
   2>&1 | tee "${OUTPUT_DIR}/deepspeed_ppo.log"

DS_EXIT_CODE=$?

# Run FSDP2 PPO
echo ""
echo "Running FSDP2 PPO backend..."
python3 -m openrlhf.cli.train_ppo_ray \
   --backend fsdp2 \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node ${NUM_GPUS} \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node ${NUM_GPUS} \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node ${NUM_GPUS} \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node ${NUM_GPUS} \
   --vllm_num_engines 2 \
   --vllm_tensor_parallel_size 1 \
   --colocate_all_models \
   --vllm_gpu_memory_utilization 0.3 \
   --pretrain ${MODEL} \
   --reward_pretrain ${REWARD_MODEL} \
   --save_path "${OUTPUT_DIR}/fsdp2_ppo" \
   --ckpt_path "${OUTPUT_DIR}/fsdp2_ppo_ckpt" \
   --train_batch_size ${TRAIN_BATCH_SIZE} \
   --rollout_batch_size ${ROLLOUT_BATCH_SIZE} \
   --micro_train_batch_size ${MICRO_BATCH_SIZE} \
   --n_samples_per_prompt 1 \
   --max_epochs ${MAX_EPOCHS} \
   --prompt_max_len ${PROMPT_MAX_LEN} \
   --generate_max_len ${GENERATE_MAX_LEN} \
   --max_samples ${MAX_SAMPLES} \
   --zero_stage 2 \
   --param_dtype bf16 \
   --actor_learning_rate ${LR} \
   --critic_learning_rate ${LR} \
   --init_kl_coef 0.01 \
   --prompt_data Open-Orca/OpenOrca \
   --input_key question \
   --normalize_reward \
   --gradient_checkpointing \
   --vllm_sync_backend nccl \
   --enforce_eager \
   --seed ${SEED} \
   --logging_steps 1 \
   2>&1 | tee "${OUTPUT_DIR}/fsdp2_ppo.log"

FSDP2_EXIT_CODE=$?

# Extract and compare metrics
echo ""
echo "======================================"
echo "Extracting metrics..."
echo "======================================"

# Extract reward and KL values from logs
echo "DeepSpeed metrics:"
grep -E "(reward|kl)" "${OUTPUT_DIR}/deepspeed_ppo.log" | tail -20

echo ""
echo "FSDP2 metrics:"
grep -E "(reward|kl)" "${OUTPUT_DIR}/fsdp2_ppo.log" | tail -20

echo ""
echo "======================================"
echo "Results Summary"
echo "======================================"
echo "DeepSpeed exit code: ${DS_EXIT_CODE}"
echo "FSDP2 exit code: ${FSDP2_EXIT_CODE}"
echo "Logs saved in ${OUTPUT_DIR}"
echo "======================================"
