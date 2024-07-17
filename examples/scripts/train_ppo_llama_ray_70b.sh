set -x 

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json='{"working_dir": "/openrlhf"}' \
    --no-wait \
    -- python3 -m openrlhf.cli.train_ppo_ray \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 2 \
    --reward_num_nodes 1 \
    --reward_num_gpus_per_node 2 \
    --critic_num_nodes 1 \
    --critic_num_gpus_per_node 4 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 4 \
    --vllm_num_engines 2 \
    --vllm_tensor_parallel_size 2 \
    --pretrain meta-llama/Meta-Llama-3-70B-Instruct \
    --reward_pretrain meta-llama/Meta-Llama-3-70B-Instruct \
    --save_path /openrlhf/examples/checkpoint/llama-3-70b-rlhf \
    --micro_train_batch_size 1 \
    --train_batch_size 128 \
    --micro_rollout_batch_size 2 \
    --rollout_batch_size 1024 \
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
    --max_samples 10000 \
    --normalize_reward \
    --adam_offload \
    --flash_attn \
    --gradient_checkpointing

