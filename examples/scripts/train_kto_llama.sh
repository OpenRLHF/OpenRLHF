set -x 

read -r -d '' training_commands <<EOF
../train_kto.py \
   --save_path ./checkpoint/llama3-8b-kto \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 256 \
   --micro_train_batch_size 1 \
   --pretrain OpenLLMAI/Llama-3-8b-sft-mixture \
   --bf16 \
   --max_epochs 1 \
   --max_len 8192 \
   --zero_stage 3 \
   --learning_rate 9e-6 \
   --dataset OpenLLMAI/preference_dataset_mixture2_and_safe_pku \
   --apply_chat_template \
   --chosen_key chosen \
   --rejected_key rejected \
   --flash_attn \
   --beta 0.1 \
   --gradient_checkpointing \
   --vanilla_loss \
EOF
     # --wandb [WANDB_TOKENS] or True (use wandb login command)
     # --vanilla_loss [for same num +/- samples in KTO batch]

# support unpaired-preference dataset, like the following:
# --dataset Dylan2048/ultrafeedback-unpaired-preferences \
# --dataset_probs 1.0 \
# --unpaired_preference


if [[ ${1} != "slurm" ]]; then
    deepspeed $training_commands
fi
