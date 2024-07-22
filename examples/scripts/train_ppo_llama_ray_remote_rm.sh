set -x
export PATH=$HOME/.local/bin/:$PATH

# tiny llama for dev
pretrain_path=TinyLlama-1.1B-Chat-v0.2
critic_pretrain_path=./ckpt/tiny_llama/tiny_llama_rm
reward_model_path=./ckpt/tiny_llama/tiny_llama_rm

save_path=./ckpt/tiny_llama
remote_rm_url=http://127.0.0.1:5000/get_rm_score

prompt_data=yahma/alpaca-cleaned,Dahoas/full-hh-rlhf,tasksource/oasst1_pairwise_rlhf_reward
prompt_data_probs=0.3,0.6,0.1

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json='{"working_dir": "/openrlhf"}' \
    --no-wait \
    -- python3 -m openrlhf.cli.train_ppo_ray \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 1 \
    --reward_num_nodes 1 \
    --reward_num_gpus_per_node 1 \
    --critic_num_nodes 1 \
    --critic_num_gpus_per_node 2 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 4 \
    --vllm_num_engines 0 \
    --vllm_tensor_parallel_size 2 \
    --pretrain ${pretrain_path} \
    --reward_pretrain ${reward_model_path} \
    --remote_rm_url ${remote_rm_url} \
    --critic_pretrain ${critic_pretrain_path} \
    --save_path ${save_path} \
    --logging_steps 1 \
    --micro_train_batch_size 1 \
    --train_batch_size 128 \
    --micro_rollout_batch_size 2 \
    --rollout_batch_size 16 \
    --max_epochs 1 \
    --prompt_max_len 1024 \
    --generate_max_len 1024 \
    --zero_stage 2 \
    --bf16 \
    --actor_learning_rate 5e-7 \
    --critic_learning_rate 9e-6 \
    --init_kl_coef 0.01 \
    --prompt_data ${prompt_data} \
    --prompt_data_probs ${prompt_data_probs} \
    --max_samples 80000 \
    --normalize_reward \
    --adam_offload \
    --flash_attn \
    --gradient_checkpointing