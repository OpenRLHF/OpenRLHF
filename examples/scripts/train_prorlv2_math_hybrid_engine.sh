#!/bin/bash
# ProRL v2: Prolonged Reinforcement Learning for LLM Reasoning
# Reference: https://developer.nvidia.com/blog/scaling-llm-reinforcement-learning-with-prolonged-training-using-prorl-v2/
#
# Key techniques:
# - REINFORCE++-baseline with batch advantage normalization
# - Clip-Higher (--actor.eps_clip_low_high 0.2 0.27) for exploration
# - Dynamic Sampling (--dynamic_filtering) to reduce noise
# - KL-regularized trust regions (--algo.kl.use_loss --algo.kl.estimator k2)
# - TIS/ICEPOP/MIS (--algo.advantage.is_correction_type) for importance sampling correction
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
   --ref.num_nodes 1 \
   --ref.num_gpus_per_node 8 \
   --actor.num_nodes 1 \
   --actor.num_gpus_per_node 8 \
   --vllm.num_engines 8 \
   --vllm.tensor_parallel_size 1 \
   --train.colocate_all \
   --vllm.gpu_memory_utilization 0.75 \
   --algo.kl.init_coef 1e-4 \
   --algo.advantage.gamma 1.0 \
   --algo.kl.use_loss \
   --algo.kl.estimator k2 \
   --algo.advantage.estimator reinforce_baseline \
   --algo.dynamic_filtering_enable \
   --algo.dynamic_filtering_range 0 1 \
   --actor.eps_clip_low_high 0.2 0.27 \
   --actor.model_name_or_path ${MODEL_PATH} \
   --reward.remote_url ${REWARD_FUNC_PATH} \
   --ckpt.output_dir ${SAVE_PATH} \
   --ckpt.path "${SAVE_PATH}/ckpt" \
   --ckpt.save_steps 5 \
   --ckpt.save_hf \
   --train.batch_size 1024 \
   --rollout.batch_size 512 \
   --rollout.n_samples_per_prompt 16 \
   --train.dynamic_batch_enable \
   --train.num_episodes 100 \
   --data.max_len 9216 \
   --rollout.max_new_tokens 8192 \
   --ds.zero_stage 3 \
   --ds.param_dtype bf16 \
   --actor.adam.lr 1e-6 \
   --data.prompt_dataset ${DATASET_PATH} \
   --data.input_key prompt \
   --data.label_key label \
   --eval.dataset OpenRLHF/aime-2024 \
   --eval.steps 5 \
   --eval.temperature 1.0 \
   --eval.n_samples_per_prompt 4 \
   --data.apply_chat_template \
   --actor.gradient_checkpointing_enable \
   --ds.packing_samples \
   --vllm.sync_backend nccl \
   --vllm.enforce_eager \
   --vllm.enable_sleep \
   --ds.enable_sleep \
   --algo.advantage.is_correction_enable \
   --algo.advantage.is_correction_threshold 0.5 5.0 \
   --algo.advantage.is_correction_type icepop \
   --train.max_tokens_per_gpu 32768 \
   --reward.stop_properly_penalty_coef 0.0

# ProRL v2 Key Parameters:
#
# REINFORCE++-baseline with batch advantage normalization:
#   --algo.advantage.estimator reinforce_baseline
#
# TIS/ICEPOP/MIS (Importance Sampling Correction):
#   --algo.advantage.is_correction_enable: Enable vLLM importance sampling correction for off-policy rollouts
#   --algo.advantage.is_correction_threshold 0.5 5.0: IS truncation interval [low, high]
#   --algo.advantage.is_correction_type icepop: Set IS coefficients outside [low, high] to 0 (instead of clamp)
#
# Length Penalty (Two options, can be used together):
#
# Option 1: Overlong Penalty (Scheduled Cosine Length Penalty based on response length)
#   --reward.overlong_buffer_len 6144: Buffer length before max, penalty starts when response > (max_new_tokens - overlong_buffer_len)
#   --reward.overlong_penalty_factor 1.0: Maximum penalty factor for overlong outputs
#   Formula: penalty = -min(exceed_len, buffer_len) / buffer_len * penalty_factor
#
# Option 2: Stop Properly Penalty (based on vLLM finish_reason == "length")
#   --reward.stop_properly_penalty_coef 0.0: Penalty coefficient [0,1] for truncated samples
#   Truncated sample rewards are scaled by this coefficient (0.0 = zero reward for truncated)
#   This encourages the model to generate complete responses within max_tokens limit
#
# Additional options you may try:
#   --train.async_enable                    # Enable async training for higher throughput