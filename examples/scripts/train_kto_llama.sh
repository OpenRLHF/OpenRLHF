set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_kto \
   --save_path ./checkpoint/llama3-8b-kto \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 64 \
   --micro_train_batch_size 1 \
   --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
   --bf16 \
   --max_epochs 1 \
   --max_len 8192 \
   --zero_stage 3 \
   --learning_rate 5e-7 \
   --dataset Dylan2048/ultrafeedback-unpaired-preferences \
   --input_key instruction \
   --output_key response \
   --label_key score \
   --flash_attn \
   --beta 0.1 \
   --max_samples 1024 \
   --gradient_checkpointing
EOF
    # --use_wandb [WANDB_TOKENS] or True (use wandb login command)


if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi
