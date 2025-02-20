set -x 

# python -m openrlhf.cli.serve_rm \
#     --reward_pretrain OpenRLHF/Llama-3-8b-rm-700k \
#     --port 5000 \
#     --bf16 \
#     --flash_attn \
#     --normalize_reward \
#     --max_len 8192 \
#     --batch_size 16

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/openrlhf"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 2 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 2 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 2 \
   --vllm_num_engines 2 \
   --vllm_tensor_parallel_size 2 \
   --colocate_actor_ref \
   --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
   --remote_rm_url http://localhost:5000/get_reward \
   --save_path /openrlhf/examples/checkpoint/llama3-8b-rlhf \
   --micro_train_batch_size 8 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 16 \
   --rollout_batch_size 1024 \
   --max_samples 100000 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --generate_max_len 1024 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.01 \
   --prompt_data OpenRLHF/prompt-collection-v0.1 \
   --input_key context_messages \
   --apply_chat_template \
   --normalize_reward \
   --packing_samples \
   --adam_offload \
   --flash_attn \
   --gradient_checkpointing \
   --use_wandb {wandb_token}

