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
   --actor.model_name_or_path ${MODEL_PATH}
   # --reward.model_name_or_path ${REWARD_MODEL}
   --ckpt.load_enable

   --ckpt.output_dir ${SAVE_PATH}
   --ckpt.path "${SAVE_PATH}/ckpt"
   --ckpt.save_hf
   --ckpt.max_num 3
   --ckpt.save_steps 10
)

ROLLOUT_ARGS=(
   --train.agent_func_path ${AGENT_FUNC_PATH}
   # --reward.remote_url ${REWARD_FUNC_FILENAME}

   --data.prompt_dataset ${DATASET_PATH}
   --data.input_key prompt
   --data.label_key label
   --data.max_len 74240
   --rollout.max_new_tokens 64000
   --data.apply_chat_template
   --ds.packing_samples

   --rollout.batch_size 128
   --rollout.n_samples_per_prompt 8
   --train.batch_size 1024
   --algo.dynamic_filtering_enable
   --algo.dynamic_filtering_range 0.0 1.0

   --train.dynamic_batch_enable
   --train.max_tokens_per_gpu 16192
   --rollout.max_tokens_per_gpu 32768

   --train.micro_batch_size 1
   --rollout.micro_batch_size 8
   --data.max_samples 128000
   --train.max_epochs 1
   --train.num_episodes 1
)

ENGINE_ARGS=(
   --train.async_enable
   --train.partial_rollout_enable

   --ref.num_nodes 1
   --ref.num_gpus_per_node 4
   --actor.num_nodes 1
   --actor.num_gpus_per_node 4
   --vllm.num_engines 2
   --vllm.tensor_parallel_size 2
   --vllm.gpu_memory_utilization 0.95
   --train.colocate_all
   --ds.enable_sleep
   --vllm.sync_backend nccl
   --vllm.enforce_eager

   --ds.zero_stage 3
   --actor.gradient_checkpointing_enable
   # --ds.adam_offload
   --ds.ring_attn_size 2
   --ds.ring_attn_head_stride 2
   --ds.param_dtype bf16
)

OPTIMIZER_ARGS=(
   --algo.advantage.estimator reinforce_baseline
   --actor.adam.lr 5e-7
   # --critic.adam.lr 9e-6
   --actor.entropy_coef 0.0
   --algo.kl.init_coef 1e-5
   --algo.kl.use_loss
   --algo.kl.estimator k2
   --algo.advantage.is_correction_enable
   --algo.advantage.is_correction_type icepop
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
