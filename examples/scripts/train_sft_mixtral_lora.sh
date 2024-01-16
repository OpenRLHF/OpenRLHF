set -x

read -r -d '' training_commands <<EOF
../train_sft.py \
    --max_len 2048 \
    --dataset Open-Orca/OpenOrca \
    --dataset_probs 1.0 \
    --train_batch_size 128 \
    --micro_train_batch_size 1 \
    --max_samples 500000 \
    --pretrain mistralai/Mixtral-8x7B-v0.1 \
    --save_path ./ckpt/mixtral_sft\
    --save_steps -1 \
    --logging_steps 1 \
    --eval_steps -1 \
    --zero_stage 3 \
    --max_epochs 1 \
    --bf16 \
    --flash_attn \
    --learning_rate 5e-6 \
    --gradient_checkpointing \
    --lora_rank 64 \
    --lora_alpha 16 \
    --disable_trace_cache \
    --aux_loss_coef 0.001
EOF

if [[ ${1} != "slurm" ]]; then
    export PATH=$HOME/.local/bin/:$PATH
    deepspeed $training_commands
fi