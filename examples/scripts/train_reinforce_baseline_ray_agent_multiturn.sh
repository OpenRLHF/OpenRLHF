set -x

NUM_GPUS=${1:-8}

# Please install gem from the following link
# pip install git+https://github.com/axon-rl/gem.git

export OPENRLHF_ASYNC_NUM_TASKS=128

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/openrlhf"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node $((NUM_GPUS)) \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node $((NUM_GPUS)) \
   --vllm_num_engines $((NUM_GPUS)) \
   --vllm_tensor_parallel_size 1 \
   --colocate_all_models \
   --agent_func_path /openrlhf/examples/python/agent_func_gem_multiturn.py \
   --pretrain Qwen/Qwen3-1.7B \
   --save_path /openrlhf/examples/test_scripts/checkpoint/llama3-8b-rlhf-rf_baseline \
   --vllm_gpu_memory_utilization 0.8 \
   --l2 0.01 \
   --micro_train_batch_size 8 \
   --train_batch_size 256 \
   --micro_rollout_batch_size 16 \
   --rollout_batch_size 128  \
   --n_samples_per_prompt 16 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --max_samples 1000000 \
   --num_episodes 1000000 \
   --generate_max_len 10240 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --advantage_estimator reinforce_baseline \
   --init_kl_coef 0 \
   --kl_estimator k2 \
   --prompt_data OpenRLHF/gem_guess_game \
   --input_key query \
   --label_key label \
   --normalize_reward \
   --vllm_enable_sleep \
   --use_dynamic_batch \
   --deepspeed_enable_sleep \
   --gradient_checkpointing \
   --packing_samples \
   --save_steps -1 \
   --enforce_eager  \
   --eps_clip_low_high 0.2 0.28 \
   --enable_prefix_caching \
   --top_p 0.98 \
   --enable_vllm_is_correction

   # --dynamic_filtering \
   # --load_checkpoint \
   # --save_steps 2 \
   # --ckpt_path /openrlhf/examples/test_scripts/checkpoint/llama3-8b-rlhf-reinforce
