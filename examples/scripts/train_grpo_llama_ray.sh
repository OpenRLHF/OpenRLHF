set -x

# reinforce++

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/openrlhf"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 1 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 1 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 4 \
   --vllm_num_engines 2 \
   --vllm_tensor_parallel_size 1 \
   --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
   --reward_pretrain OpenRLHF/Llama-3-8b-rm-700k \
   --save_path /openrlhf/examples/test_scripts/checkpoint/llama3-8b-rlhf \
   --micro_train_batch_size 16 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 32 \
   --rollout_batch_size 1024 \
   --n_samples_per_prompt 1 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --max_samples 100000 \
   --generate_max_len 1024 \
   --init_kl_coef 1e-3 \
   --gamma 1.0 \
   --use_kl_loss \
   --kl_estimator k3 \
   --advantage_estimator group_norm \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --prompt_data OpenRLHF/prompt-collection-v0.1 \
   --input_key context_messages \
   --apply_chat_template \
   --normalize_reward \
   --adam_offload \
   --gradient_checkpointing \
   --packing_samples \
   --save_steps -1 \
   --ckpt_path /openrlhf/examples/test_scripts/ckpt/llama3-8b-rlhf

# You could also try
#   --kl_estimator k2 \

# also supports --advantage_estimator rloo | reinforce_baseline
