#!/bin/bash

# Example script for PPO training with streaming asynchronous sampling.
# Implements prompt-level asynchronous gathering based on the OpenRLHF Agent architecture.

set -x

# Model and data configuration
export PRETRAIN="microsoft/DialoGPT-medium"
export REWARD_MODEL="microsoft/DialoGPT-medium"
export DATASET="Anthropic/hh-rlhf"

# Ray cluster configuration (if applicable)
export RAY_CLUSTER_CONFIG="examples/scripts/ray_cluster.yaml"

# Asynchronous sampling configuration
export OPENRLHF_ASYNC_NUM_TASKS=128  # Number of concurrent agents
export OPENRLHF_ASYNC_QUEUE_SIZE=2   # Controls the degree of off-policy learning

# GPU and parallelism configuration
ACTOR_NUM_NODES=1
ACTOR_NUM_GPUS_PER_NODE=4
CRITIC_NUM_NODES=1
CRITIC_NUM_GPUS_PER_NODE=4
VLLM_NUM_ENGINES=2
VLLM_TENSOR_PARALLEL_SIZE=2

# Batch and sampling configuration
ROLLOUT_BATCH_SIZE=192
N_SAMPLES_PER_PROMPT=8
MICRO_ROLLOUT_BATCH_SIZE=8

# Streaming sampling and dynamic filtering configuration
ENABLE_STREAMING_SAMPLING=true
DYNAMIC_FILTERING=true
DYNAMIC_FILTERING_REWARD_RANGE="0.1 0.9"

# Agent configuration (for multi-step agent interaction)
AGENT_FUNC_PATH="examples/batch_inference/agent_func.py"

python -m openrlhf.cli.train_ppo_ray \
    --pretrain ${PRETRAIN} \
    --reward_pretrain ${REWARD_MODEL} \
    --dataset ${DATASET} \
    --dataset_probs 1.0 \
    --save_path ./ckpt/streaming_ppo_model \
    --save_steps 100 \
    --logging_steps 1 \
    --eval_steps 100 \
    --ckpt_path ./ckpt/streaming_ppo_model \
    --max_ckpt_num 3 \
    \
    --rollout_batch_size ${ROLLOUT_BATCH_SIZE} \
    --micro_rollout_batch_size ${MICRO_ROLLOUT_BATCH_SIZE} \
    --n_samples_per_prompt ${N_SAMPLES_PER_PROMPT} \
    --train_batch_size 128 \
    --micro_train_batch_size 4 \
    --max_epochs 1 \
    --prompt_max_len 512 \
    --generate_max_len 512 \
    \
    --actor_num_nodes ${ACTOR_NUM_NODES} \
    --actor_num_gpus_per_node ${ACTOR_NUM_GPUS_PER_NODE} \
    --critic_num_nodes ${CRITIC_NUM_NODES} \
    --critic_num_gpus_per_node ${CRITIC_NUM_GPUS_PER_NODE} \
    \
    --vllm_num_engines ${VLLM_NUM_ENGINES} \
    --vllm_tensor_parallel_size ${VLLM_TENSOR_PARALLEL_SIZE} \
    --vllm_gpu_memory_utilization 0.85 \
    --vllm_sync_backend nccl \
    \
    --async_train \
    --dynamic_filtering \
    --dynamic_filtering_reward_range ${DYNAMIC_FILTERING_REWARD_RANGE} \
    --enable_streaming_sampling \
    \
    --agent_func_path ${AGENT_FUNC_PATH} \
    \
    --actor_learning_rate 1e-6 \
    --critic_learning_rate 9e-6 \
    --adam_offload \
    --flash_attn \
    --gradient_checkpointing \
    --use_wandb [your_wandb_api_key] \
    --wandb_project "openrlhf_streaming_sampling" \
    --wandb_run_name "streaming_ppo_${ROLLOUT_BATCH_SIZE}_${N_SAMPLES_PER_PROMPT}"

echo "🏁 Streaming sampling PPO training completed!"