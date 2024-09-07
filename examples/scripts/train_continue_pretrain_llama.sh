set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
   --max_len 4096 \
   --input_key text \
   --dataset DKYoon/SlimPajama-6B \
   --pretrain_mode \
   --train_batch_size 128 \
   --micro_train_batch_size 4 \
   --max_samples 1000000 \
   --pretrain meta-llama/Meta-Llama-3.1-8B-Instruct \
   --save_path ./checkpoint/llama3.1-8b-cft \
   --save_steps 5000 \
   --logging_steps 100 \
   --eval_steps 500 \
   --zero_stage 3 \
   --max_epochs 1 \
   --bf16 \
   --flash_attn \
   --learning_rate 3e-4 \
   --gradient_checkpointing \
   --packing_samples
EOF
    # --wandb [WANDB_TOKENS]

if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi