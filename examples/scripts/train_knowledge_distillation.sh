set -x 

read -r -d '' training_commands <<EOF
../train_kd.py \
    --max_len 2048 \
    --dataset Open-Orca/OpenOrca \
    --dataset_probs 1.0 \
    --train_batch_size 128 \
    --micro_train_batch_size 1 \
    --max_samples 50000 \
    --pretrain meta-llama/Llama-2-7b-hf \
    --teacher_model meta-llama/Llama-2-13b-chat-hf \
    --save_path ./ckpt/llama2 \
    --save_steps -1 \
    --logging_steps 1 \
    --eval_steps -1 \
    --zero_stage 3 \
    --max_epochs 1 \
    --bf16 \
    --flash_attn \
    --kd_coef 0.4 \
    --learning_rate 5e-6 \
    --gradient_checkpointing 
EOF
     # --wandb [WANDB_TOKENS] or True (use wandb login command)


if [[ ${1} != "slurm" ]]; then
    deepspeed $training_commands
fi