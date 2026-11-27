#!/bin/bash
# FlashREINFORCE: critic-free, single-rollout asynchronous RL for agentic language models
# (paper: https://www.researchgate.net/publication/414274571_FlashREINFORCE_FLASHREINFORCE_CRITIC-FREE_SINGLE-ROLLOUT_ASYNCHRONOUS_RL_FOR_AGENTIC_LANGUAGE_MODELS).
#
# No new loss: with one optimizer step per rollout batch (--train.max_epochs 1, --train.batch_size ==
# --rollout.batch_size) the PPO ratio is 1, so the surrogate is the REINFORCE gradient and the IS
# correction supplies the pi/mu weight against the vLLM behavior logprobs. The algorithm is configuration:
#   --rollout.n_samples_per_prompt 1                  one rollout per prompt
#   --algo.advantage.estimator flash_reinforce        reward minus the rollout-batch mean, no group, no whitening
#   --algo.advantage.is_correction_level seq          IS weight pi/mu against the vLLM behavior logprobs,
#   --algo.advantage.is_correction_gating binary_kl     gated per sequence by the mean sampled-token
#   --algo.advantage.is_correction_threshold 5e-3       binary KL (two-sided trust region, delta = 5e-3)
#   --actor.loss_agg_mode seq-mean-token-mean         sample mean: every rollout weighs the same
#   --algo.kl.init_coef 0                             no KL penalty, no reference model
# The paper's FP16 sanity test runs DeepSeek-R1-Distill-Qwen-1.5B on sail/Sanity-Test-R1D-1.5B (1,460 MATH
# problems, 8k responses, 128 rollouts per update, lr 1e-6, wd 0.1, temperature 1.0) for 12,000 updates;
# 512 rollouts stay in flight and up to 8 finished batches queue ahead of the trainer. 8 GPUs: 1 actor + 7 vLLM.

SCRIPT_DIR="$(dirname "$0")"
WORK_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)

set -x

MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DATASET_PATH="zhuzilin/dapo-math-17k"
SAVE_PATH="${WORK_DIR}/exp/flash-reinforce-r1d-1.5b"
REWARD_FUNC_PATH="examples/python/math_reward_func.py"

CKPT_ARGS=(
   --actor.model_name_or_path ${MODEL_PATH}
   --ckpt.load_enable

   --ckpt.output_dir ${SAVE_PATH}
   --ckpt.path "${SAVE_PATH}/ckpt"
   --ckpt.save_hf
   --ckpt.max_num 3
   --ckpt.save_steps 50
)

ROLLOUT_ARGS=(
   --reward.remote_url ${REWARD_FUNC_PATH}

   --data.prompt_dataset ${DATASET_PATH}
   --data.input_key prompt
   --data.label_key label
   --data.apply_chat_template
   --data.max_len 9216
   --rollout.max_new_tokens 8192
   --rollout.temperature 1.0
   --rollout.top_p 1.0
   --ds.packing_samples

   --rollout.batch_size 128
   --rollout.vllm_generate_batch_size 512
   --rollout.n_samples_per_prompt 1
   --rollout.micro_batch_size 1
   --train.batch_size 128
   --train.micro_batch_size 1
   --train.max_epochs 1
   --train.num_episodes 1000
   --data.max_samples 128000
)

ENGINE_ARGS=(
   --train.async_enable
   --train.async_queue_size 8

   --actor.num_nodes 1
   --actor.num_gpus_per_node 1
   --vllm.num_engines 7
   --vllm.tensor_parallel_size 1
   --vllm.gpu_memory_utilization 0.9

   --ds.zero_stage 2
   --actor.gradient_checkpointing_enable
   --ds.param_dtype bf16
)

OPTIMIZER_ARGS=(
   --algo.advantage.estimator flash_reinforce
   --algo.advantage.is_correction_level seq
   --algo.advantage.is_correction_gating binary_kl
   --algo.advantage.is_correction_threshold 5e-3
   --actor.loss_agg_mode seq-mean-token-mean
   --algo.kl.init_coef 0
   --actor.adam.lr 1e-6
   --actor.adam.weight_decay 0.1
)

LOG_ARGS=(
   --logger.tensorboard_dir ${SAVE_PATH}/runs
   --logger.logging_steps 1
   --eval.steps -1
)

ray job submit --address="http://127.0.0.1:8265" \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${ENGINE_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${LOG_ARGS[@]}
