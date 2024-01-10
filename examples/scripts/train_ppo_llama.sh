set -x 

read -r -d '' training_commands <<EOF
../train_ppo.py \
    --pretrain OpenLLMAI/Llama-2-7b-sft-model-ocra-500k \
    --reward_pretrain OpenLLMAI/Llama-2-7b-rm-anthropic_hh-lmsys-oasst-webgpt \
    --save_path ./ckpt/7b_llama \
    --save_steps -1 \
    --logging_steps 1 \
    --eval_steps -1 \
    --micro_train_batch_size 2 \
    --train_batch_size 128 \
    --micro_rollout_batch_size 4 \
    --rollout_batch_size 1024 \
    --max_epochs 1 \
    --prompt_max_len 1024 \
    --generate_max_len 1024 \
    --zero_stage 2 \
    --bf16 \
    --actor_learning_rate 5e-7 \
    --critic_learning_rate 9e-6 \
    --init_kl_coef 0.01 \
    --prompt_data Open-Orca/OpenOrca,Dahoas/full-hh-rlhf,tasksource/oasst1_pairwise_rlhf_reward \
    --prompt_data_probs 0.4,0.5,0.1 \
    --max_samples 80000 \
    --normalize_reward \
    --actor_init_on_gpu \
    --adam_offload \
    --flash_attn \
    --gradient_checkpointing
EOF
    # --wandb [WANDB_TOKENS]

if [[ ${1} != "slurm" ]]; then
    export PATH=$HOME/.local/bin/:$PATH
    deepspeed $training_commands
fi
