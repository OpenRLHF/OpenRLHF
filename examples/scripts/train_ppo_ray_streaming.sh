#!/bin/bash

set -x

# Asynchronous sampling configuration
export OPENRLHF_ASYNC_NUM_TASKS=128  # Number of concurrent agents
export OPENRLHF_ASYNC_QUEUE_SIZE=2   # Controls the degree of off-policy learning

# Streaming sampling and dynamic filtering configuration
DYNAMIC_FILTERING_REWARD_RANGE="0.1 0.9"
# Agent configuration (for multi-step agent interaction)
AGENT_FUNC_PATH="examples/python/agent_func.py"

python3 -m openrlhf.cli.train_ppo_ray \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 8 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 8 \
    --vllm_num_engines 4 \
    --vllm_tensor_parallel_size 2 \
    --colocate_all_models \
    --vllm_gpu_memory_utilization 0.6 \
    \
    --init_kl_coef 1e-3 \
    --gamma 1.0 \
    --use_kl_loss \
    --kl_estimator k3 \
    --advantage_estimator reinforce_baseline \
    \
    --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
    --reward_pretrain OpenRLHF/Llama-3-8b-rm-700k \
    --save_path /openrlhf/examples/test_scripts/final/llama3-8b-rlhf \
    --ckpt_path /openrlhf/examples/test_scripts/ckpt/llama3-8b-rlhf \
    --save_hf_ckpt \
    \
    --prompt_data OpenRLHF/prompt-collection-v0.1 \
    --input_key context_messages \
    --apply_chat_template \
    \
    --micro_train_batch_size 8 \
    --train_batch_size 128 \
    --micro_rollout_batch_size 16 \
    --rollout_batch_size 128 \
    --n_samples_per_prompt 8 \
    --max_epochs 1 \
    --prompt_max_len 1024 \
    --max_samples 100000 \
    --generate_max_len 1024 \
    --zero_stage 3 \
    --bf16 \
    --actor_learning_rate 5e-7 \
    --critic_learning_rate 9e-6 \
    \
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
    --gradient_checkpointing \
    --packing_samples \
    --vllm_sync_backend nccl \
    --enforce_eager \
    --vllm_enable_sleep \
    --deepspeed_enable_sleep \
    --async_train \
    --dynamic_filtering \
    --dynamic_filtering_reward_range ${DYNAMIC_FILTERING_REWARD_RANGE} \
    --enable_streaming_sampling \
    --agent_func_path ${AGENT_FUNC_PATH}
