set -x

# pip install mamba-ssm causal-conv1d>=1.2.0

read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
    --max_len 8192 \
    --dataset Open-Orca/OpenOrca \
    --input_key question \
    --output_key response \
    --train_batch_size 128 \
    --micro_train_batch_size 4 \
    --pretrain ai21labs/Jamba-v0.1 \
    --save_path ./checkpoint/jamba-sft-lora\
    --save_steps -1 \
    --logging_steps 1 \
    --eval_steps -1 \
    --zero_stage 3 \
    --max_epochs 2 \
    --bf16 \
    --gradient_checkpointing \
    --flash_attn \
    --learning_rate 5e-6 \
    --lora_rank 64 \
    --lora_alpha 64 \
    --aux_loss_coef 0.001
EOF

if [[ ${1} != "slurm" ]]; then
    
    deepspeed --module $training_commands
fi