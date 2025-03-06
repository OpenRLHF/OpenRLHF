set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_ppo \
   --pretrain Qwen/Qwen2.5-3B \
   --reward_pretrain Qwen/Qwen2.5-0.5B \
   --save_path ./checkpoint/qwen-2.5 \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --micro_train_batch_size 2 \
   --train_batch_size 8 \
   --micro_rollout_batch_size 2 \
   --rollout_batch_size 8 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --generate_max_len 1024 \
   --zero_stage 2 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.01 \
   --prompt_data OpenRLHF/prompt-collection-v0.1 \
   --input_key context_messages \
   --apply_chat_template \
   --lora_rank 4 \
   --lora_alpha 4 \
   --use_lora_disable \
   --max_samples 16 \
   --normalize_reward \
   --adam_offload \
   --load_checkpoint \
   --gradient_checkpointing
EOF

    # --packing_samples
    # --use_wandb [WANDB_TOKENS] or True (use wandb login command)
    # --remote_rm_url http://localhost:5000/get_reward

if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi