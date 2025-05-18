set -x

# reinforce++-baseline

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/openrlhf"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 6 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 6 \
   --vllm_num_engines 2 \
   --vllm_tensor_parallel_size 1 \
   --colocate_all_models \
   --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
   --agent_func_path /openrlhf/examples/python/agent_func.py \
   --save_path /openrlhf/examples/test_scripts/checkpoint/llama3-8b-rlhf \
   --micro_train_batch_size 8 \
   --train_batch_size 192 \
   --micro_rollout_batch_size 16 \
   --rollout_batch_size 192 \
   --n_samples_per_prompt 8 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --max_samples 12500 \
   --generate_max_len 1024 \
   --advantage_estimator reinforce_baseline \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --init_kl_coef 1e-3 \
   --use_kl_loss \
   --kl_estimator k2 \
   --prompt_data OpenRLHF/prompt-collection-v0.1 \
   --input_key context_messages \
   --apply_chat_template \
   --normalize_reward \
   --gradient_checkpointing \
   --packing_samples \
   --save_steps -1 \
   --async_train \
   --ckpt_path /openrlhf/examples/test_scripts/ckpt/llama3-8b-rlhf

# --colocate_all_models with --async_train only merge the deepspeed models, not the vllm engines

# You could also try
#   --use_kl_loss \
#   --kl_estimator k3 | k2 \

# also supports --advantage_estimator rloo | reinforce_baseline
