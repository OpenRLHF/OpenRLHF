set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_dpo \
     --save_path ./checkpoint/llama2-34b-dpo \
     --save_steps -1 \
     --logging_steps 1 \
     --eval_steps -1 \
     --train_batch_size 128 \
     --micro_train_batch_size 1 \
     --pretrain codellama/CodeLlama-34b-Instruct-hf \
     --bf16 \
     --max_epochs 1 \
     --max_len 2048 \
     --zero_stage 3 \
     --beta 0.1 \
     --learning_rate 5e-7 \
     --dataset OpenRLHF/preference_dataset_mixture2_and_safe_pku \
     --apply_chat_template \
     --chosen_key chosen \
     --rejected_key rejected \
     --flash_attn \
     --gradient_checkpointing \
     --adam_offload
EOF
     # --use_wandb [WANDB_TOKENS] or True (use wandb login command)
     # --ipo [for IPO]
     # --label_smoothing 0.1 [for cDPO]


if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi
