set -x 
export PATH=$HOME/.local/bin/:$PATH

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json='{"working_dir": "/openrlhf", "pip": "/openrlhf/requirements.txt"}' \
    --no-wait \
    -- python3 examples/train_ppo_ray.py \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 1 \
    --reward_num_nodes 1 \
    --reward_num_gpus_per_node 1 \
    --critic_num_nodes 1 \
    --critic_num_gpus_per_node 2 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 4 \
    --pretrain codellama/CodeLlama-34b-Instruct-hf \
    --reward_pretrain codellama/CodeLlama-34b-Instruct-hf \
    --save_path /openrlhf/examples/test_scripts/ckpt/34b_codellama \
    --micro_train_batch_size 1 \
    --train_batch_size 128 \
    --micro_rollout_batch_size 2 \
    --rollout_batch_size 1024 \
    --max_epochs 1 \
    --prompt_max_len 1024 \
    --generate_max_len 1024 \
    --zero_stage 2 \
    --bf16 \
    --actor_learning_rate 5e-7 \
    --critic_learning_rate 9e-6 \
    --init_kl_coef 0.01 \
    --prompt_data Open-Orca/OpenOrca,Dahoas/full-hh-rlhf,tasksource/oasst1_pairwise_rlhf_reward \
    --prompt_data_probs 0.4,0.5,0.1 \
    --max_samples 80000 \
    --normalize_reward \
    --actor_init_on_gpu \
    --adam_offload \
    --flash_attn \
    --gradient_checkpointing