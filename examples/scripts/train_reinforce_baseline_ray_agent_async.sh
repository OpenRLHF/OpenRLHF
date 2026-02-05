#!/bin/bash

SCRIPT_DIR="$(dirname "$0")"
WORK_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)

set -x

MODEL_PATH="Qwen/Qwen3-4B-Thinking-2507"
DATASET_PATH="zhuzilin/dapo-math-17k"
SAVE_PATH="${WORK_DIR}/exp/Qwen3-4B-Thinking"

# For test
AGENT_FUNC_PATH="examples/python/agent_func.py"

# For demo
# git clone https://github.com/Freder-chen/OpenRLHF-Agent.git
# cd OpenRLHF-Agent && pip install -e .
# AGENT_FUNC_PATH="{OpenRLHF-Agent/examples/single_turn/agent_func.py}"

CKPT_ARGS=(
   --pretrain ${MODEL_PATH}
   # --reward_pretrain ${REWARD_MODEL}
   --load_checkpoint

   --save_path ${SAVE_PATH}
   --ckpt_path "${SAVE_PATH}/ckpt"
   --save_hf_ckpt
   --max_ckpt_num 3
   --save_steps 10
)

ROLLOUT_ARGS=(
   --agent_func_path ${AGENT_FUNC_PATH}
   # --remote_rm_url ${REWARD_FUNC_FILENAME}

   --prompt_data ${DATASET_PATH}
   --input_key prompt
   --label_key label
   --prompt_max_len 10240
   --generate_max_len 64000
   --apply_chat_template
   --packing_samples

   --rollout_batch_size 128
   --n_samples_per_prompt 8
   --train_batch_size 1024
   --dynamic_filtering
   --dynamic_filtering_reward_range 0.0 1.0

   --use_dynamic_batch
   --train_max_tokens_per_gpu 16192
   --rollout_max_tokens_per_gpu 32768

   --micro_train_batch_size 1
   --micro_rollout_batch_size 8
   --max_samples 128000
   --max_epochs 1
   --num_episodes 1
)

ENGINE_ARGS=(
   --async_train

   --ref_num_nodes 1
   --ref_num_gpus_per_node 4
   --actor_num_nodes 1
   --actor_num_gpus_per_node 4
   --vllm_num_engines 2
   --vllm_tensor_parallel_size 2
   --vllm_gpu_memory_utilization 0.7
   --colocate_all_models
   --deepspeed_enable_sleep
   --vllm_sync_backend nccl
   --enforce_eager

   --zero_stage 3
   --gradient_checkpointing
   # --adam_offload
   --ring_attn_size 2
   --ring_head_stride 2
   --param_dtype bf16
)

OPTIMIZER_ARGS=(
   --advantage_estimator reinforce_baseline
   --actor_learning_rate 5e-7
   # --critic_learning_rate 9e-6
   --entropy_loss_coef 0.0
   --init_kl_coef 1e-5
   --use_kl_loss
   --kl_estimator k2
   --enable_vllm_is_correction
   --vllm_is_correction_type icepop
)

LOG_ARGS=(
   --use_tensorboard ${SAVE_PATH}/runs
   --logging_steps 1
   --eval_steps -1
)

ray job submit --address="http://127.0.0.1:8265" \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${ENGINE_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${LOG_ARGS[@]}
