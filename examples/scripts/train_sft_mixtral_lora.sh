set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
    --max_len 2048 \
    --dataset Open-Orca/OpenOrca \
    --input_key question \
    --output_key response \
    --train_batch_size 128 \
    --micro_train_batch_size 4 \
    --max_samples 500000 \
    --pretrain mistralai/Mixtral-8x7B-v0.1 \
    --save_path ./checkpoint/mixtral-sft-lora\
    --save_steps -1 \
    --logging_steps 1 \
    --eval_steps -1 \
    --zero_stage 3 \
    --max_epochs 1 \
    --bf16 \
    --gradient_checkpointing \
    --attn_implementation flash_attention_2 \
    --learning_rate 5e-6 \
    --lora_rank 64 \
    --lora_alpha 64 \
    --packing_samples \
    --aux_loss_coef 0.001
EOF

if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi