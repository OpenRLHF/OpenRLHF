set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_kd \
   --max_len 2048 \
   --dataset Open-Orca/OpenOrca \
   --input_key question \
   --output_key response \
   --train_batch_size 256 \
   --micro_train_batch_size 2 \
   --max_samples 500000 \
   --pretrain meta-llama/Meta-Llama-3-8B-Instruct \
   --teacher_model meta-llama/Meta-Llama-3-70B-Instruct \
   --save_path ./checkpoint/llama3-8b-kd \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 3 \
   --max_epochs 1 \
   --bf16 \
   --flash_attn \
   --kd_coef 0.4 \
   --learning_rate 5e-6 \
   --teacher_offload \
   --gradient_checkpointing
EOF
    # --use_wandb [WANDB_TOKENS] or True (use wandb login command)


if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi
