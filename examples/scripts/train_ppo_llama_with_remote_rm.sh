set -x

pretrain_path=TinyLlama-1.1B-Chat-v0.2
critic_pretrain_path=./ckpt/tiny_llama/tiny_llama_rm
reward_model_path=./ckpt/tiny_llama/tiny_llama_rm

save_path=./ckpt/tiny_llama
remote_rm_url=http://xxx/get_rm_score

prompt_data=yahma/alpaca-cleaned,Dahoas/full-hh-rlhf,tasksource/oasst1_pairwise_rlhf_reward
prompt_data_probs=0.3,0.6,0.1

read -r -d '' training_commands <<EOF
openrlhf.cli.train_ppo \
    --pretrain ${pretrain_path} \
    --reward_pretrain ${reward_model_path} \
    --remote_rm_url ${remote_rm_url} \
    --save_path ${save_path} \
    --save_steps 10 \
    --logging_steps 2 \
    --eval_steps 1 \
    --micro_train_batch_size 1 \
    --train_batch_size 128 \
    --micro_rollout_batch_size 1 \
    --rollout_batch_size 8 \
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
    --normalize_reward \
    --adam_offload \
    --gradient_checkpointing
EOF

if [[ ${1} != "slurm" ]]; then
  export PATH=$HOME/.local/bin/:$PATH
  deepspeed --module $training_commands
fi
