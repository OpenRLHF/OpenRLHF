set -x 

read -r -d '' training_commands <<EOF
../train_kto.py \
     --save_path ./ckpt/llama_KTO_not_vanilla \
     --save_steps 5000 \
     --logging_steps 1 \
     --eval_steps -1 \
     --train_batch_size 128 \
     --micro_train_batch_size 4 \
     --pretrain OpenLLMAI/Llama-2-7b-sft-model-ocra-500k \
     --bf16 \
     --max_epochs 1 \
     --max_len 2048 \
     --zero_stage 3 \
     --beta 0.1 \
     --learning_rate 5e-7 \
     --dataset Anthropic/hh-rlhf,tasksource/oasst1_pairwise_rlhf_reward,openai/webgpt_comparisons \
     --dataset_probs 0.72,0.14,0.14 \
     --flash_attn \
     --gradient_checkpointing \
     --vanilla_loss
EOF
     # --wandb [WANDB_TOKENS]
     # --vanilla_loss [for same num +/- samples in KTO batch]

# support unpaired-preference dataset, like the following:
# --dataset Dylan2048/ultrafeedback-unpaired-preferences \
# --dataset_probs 1.0 \
# --unpaired_preference


if [[ ${1} != "slurm" ]]; then
    export PATH=$HOME/.local/bin/:$PATH
    deepspeed $training_commands
fi
