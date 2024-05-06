set -x

read -r -d '' training_commands <<EOF
../train_sft.py \
    --max_len 32000 \
    --dataset MaziyarPanahi/WizardLM_evol_instruct_V2_196k \
    --dataset_probs 1.0 \
    --train_batch_size 128 \
    --micro_train_batch_size 4 \
    --pretrain ai21labs/Jamba-v0.1 \
    --save_path ./ckpt/jamba_wizard\
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
    export PATH=$HOME/.local/bin/:$PATH
    deepspeed $training_commands
fi