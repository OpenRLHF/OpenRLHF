set -x

read -r -d '' training_commands <<EOF
../train_sft.py \
    --max_len 4096 \
    --dataset {your_pretrain_dataset} \
    --dataset_probs 1.0 \
    --train_batch_size 128 \
    --micro_train_batch_size 1 \
    --pretrain meta-llama/Llama-2-7b-hf \
    --save_path ./ckpt/7b_llama \
    --save_steps -1 \
    --logging_steps 1 \
    --eval_steps -1 \
    --zero_stage 2 \
    --max_epochs 1 \
    --bf16 \
    --flash_attn \
    --learning_rate 3e-5 \
    --gradient_checkpointing \
    --pretrain_mode
EOF

if [[ ${1} != "slurm" ]]; then
    export PATH=$HOME/.local/bin/:$PATH
    deepspeed $training_commands
fi