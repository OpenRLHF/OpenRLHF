#!/bin/bash
# ProRL v2: Prolonged Reinforcement Learning for LLM Reasoning
# Reference: https://developer.nvidia.com/blog/scaling-llm-reinforcement-learning-with-prolonged-training-using-prorl-v2/
#
# Key techniques:
# - REINFORCE++-baseline with batch advantage normalization
# - Clip-Higher (--eps_clip_low_high 0.2 0.27) for exploration
# - Dynamic Sampling (--dynamic_filtering) to reduce noise
# - KL-regularized trust regions (--use_kl_loss --kl_estimator k2)
# - TIS/ICEPOP/MIS (--vllm_is_correction_type) for importance sampling correction
# - Stop Properly Penalty (--stop_properly_penalty_coef) for truncated samples
#
# ProRL v2 achieves state-of-the-art performance among 1.5B reasoning models
# with sustained improvements across math, code, and reasoning tasks.

set -x

SCRIPT_DIR="$(dirname "$0")"
WORK_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)

# Model and Dataset Configuration
MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DATASET_PATH="OpenRLHF/dapo-math-17k"
SAVE_PATH="${WORK_DIR}/exp/DeepSeek-R1-Qwen-1.5B-PRORLV2"

# Math reward function for answer verification
REWARD_FUNC_PATH="${WORK_DIR}/examples/python/math_reward_func.py"

python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 8 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 8 \
   --vllm_num_engines 8 \
   --vllm_tensor_parallel_size 1 \
   --colocate_all_models \
   --vllm_gpu_memory_utilization 0.75 \
   --init_kl_coef 1e-4 \
   --gamma 1.0 \
   --use_kl_loss \
   --kl_estimator k2 \
   --advantage_estimator reinforce_baseline \
   --dynamic_filtering \
   --dynamic_filtering_reward_range 0 1 \
   --eps_clip_low_high 0.2 0.27 \
   --pretrain ${MODEL_PATH} \
   --remote_rm_url ${REWARD_FUNC_PATH} \
   --save_path ${SAVE_PATH} \
   --ckpt_path "${SAVE_PATH}/ckpt" \
   --save_steps 5 \
   --save_hf_ckpt \
   --train_batch_size 1024 \
   --rollout_batch_size 512 \
   --n_samples_per_prompt 16 \
   --use_dynamic_batch \
   --num_episodes 100 \
   --prompt_max_len 1024 \
   --generate_max_len 8192 \
   --zero_stage 3 \
   --param_dtype bf16 \
   --actor_learning_rate 1e-6 \
   --prompt_data ${DATASET_PATH} \
   --input_key prompt \
   --label_key label \
   --eval_dataset OpenRLHF/aime-2024 \
   --eval_steps 5 \
   --eval_temperature 1.0 \
   --eval_n_samples_per_prompt 4 \
   --apply_chat_template \
   --gradient_checkpointing \
   --packing_samples \
   --vllm_sync_backend nccl \
   --enforce_eager \
   --vllm_enable_sleep \
   --deepspeed_enable_sleep \
   --enable_vllm_is_correction \
   --vllm_is_truncated_threshold 0.5 5.0 \
   --vllm_is_correction_type icepop \
   --train_max_tokens_per_gpu 32768 \
   --stop_properly_penalty_coef 0.0

# ProRL v2 Key Parameters:
#
# REINFORCE++-baseline with batch advantage normalization:
#   --advantage_estimator reinforce_baseline
#
# TIS/ICEPOP/MIS (Importance Sampling Correction):
#   --enable_vllm_is_correction: Enable vLLM importance sampling correction for off-policy rollouts
#   --vllm_is_truncated_threshold 0.5 5.0: IS truncation interval [low, high]
#   --vllm_is_correction_type icepop: Set IS coefficients outside [low, high] to 0 (instead of clamp)
#
# Length Penalty (Two options, can be used together):
#
# Option 1: Overlong Penalty (Scheduled Cosine Length Penalty based on response length)
#   --overlong_buffer_len 6144: Buffer length before max, penalty starts when response > (generate_max_len - overlong_buffer_len)
#   --overlong_penalty_factor 1.0: Maximum penalty factor for overlong outputs
#   Formula: penalty = -min(exceed_len, buffer_len) / buffer_len * penalty_factor
#
# Option 2: Stop Properly Penalty (based on vLLM finish_reason == "length")
#   --stop_properly_penalty_coef 0.0: Penalty coefficient [0,1] for truncated samples
#   Truncated sample rewards are scaled by this coefficient (0.0 = zero reward for truncated)
#   This encourages the model to generate complete responses within max_tokens limit
#
# Additional options you may try:
#   --async_train                    # Enable async training for higher throughput