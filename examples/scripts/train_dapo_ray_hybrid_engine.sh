set -x

python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 8 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 8 \
   --vllm_num_engines 4 \
   --vllm_tensor_parallel_size 2 \
   --colocate_all_models \
   --vllm_gpu_memory_utilization 0.6 \
   --init_kl_coef 1e-3 \
   --gamma 1.0 \
   --use_kl_loss \
   --kl_estimator k3 \
   --advantage_estimator group_norm \
   --dynamic_filtering \
   --dynamic_filtering_reward_range 0 1 \
   --eps_clip_low_high 0.2 0.3 \
   --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
   --remote_rm_url /openrlhf/examples/python/reward_func.py \
   --save_path /openrlhf/examples/test_scripts/final/llama3-8b-rlhf \
   --ckpt_path /openrlhf/examples/test_scripts/ckpt/llama3-8b-rlhf \
   --save_steps 20 \
   --save_hf_ckpt \
   --micro_train_batch_size 8 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 16 \
   --rollout_batch_size 128 \
   --n_samples_per_prompt 8 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --max_samples 20000 \
   --generate_max_len 1024 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --prompt_data OpenRLHF/prompt-collection-v0.1 \
   --input_key context_messages \
   --apply_chat_template \
   --gradient_checkpointing \
   --packing_samples \
   --vllm_sync_backend nccl \
   --enforce_eager \
   --vllm_enable_sleep \
   --deepspeed_enable_sleep

# You could also try
#   --kl_estimator k2 \
