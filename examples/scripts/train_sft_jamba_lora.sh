set -x

read -r -d '' training_commands <<EOF
../train_sft.py \
    --max_len 128000 \
    --dataset MaziyarPanahi/WizardLM_evol_instruct_V2_196k \
    --dataset_probs 1.0 \
    --train_batch_size 128 \
    --micro_train_batch_size 2 \
    --pretrain /home/scratch.jianh_gpu/data/models/Jamba-v0.1 \
    --save_path ./ckpt/jamba_wizard\
    --save_steps -1 \
    --logging_steps 1 \
    --eval_steps -1 \
    --zero_stage 3 \
    --max_epochs 2 \
    --bf16 \
    --gradient_checkpointing \
    --learning_rate 5e-6 \
    --lora_rank 64 \
    --lora_alpha 16 \
    --target_modules v_proj out_proj x_proj q_proj in_proj \
    --aux_loss_coef 0.001
EOF

if [[ ${1} != "slurm" ]]; then
    export PATH=$HOME/.local/bin/:$PATH
    deepspeed $training_commands
fi