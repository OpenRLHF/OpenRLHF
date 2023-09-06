set -x 

read -r -d '' training_commands <<EOF
../train_ppo.py \
    --pretrain meta-llama/Llama-2-7b-hf \
    --critic_pretrain meta-llama/Llama-2-7b-hf \
    --reward_model_path ./ckpt/7b_llama/rm_model_anthropic_oasst_lmsys_webgpt.pt \
    --sft_model_path ./ckpt/7b_llama/sft_model_ocra.pt \
    --save_path ./ckpt/7b_llama \
    --micro_train_batch_size 1 \
    --train_batch_size 128 \
    --micro_rollout_batch_size 1 \
    --rollout_batch_size 1024 \
    --max_epochs 1 \
    --prompt_max_len 1024 \
    --generate_max_len 1024 \
    --zero_stage 2 \
    --bf16 \
    --actor_learning_rate 5e-7 \
    --critic_learning_rate 9e-6 \
    --inference_tp_size 1 \
    --init_kl_coef 0.01 \
    --prompt_data yahma/alpaca-cleaned,Dahoas/full-hh-rlhf,tasksource/oasst1_pairwise_rlhf_reward \
    --prompt_data_probs 0.3,0.6,0.1 \
    --normalize_reward \
    --actor_init_on_gpu \
    --adam_offload \
    --gradient_checkpointing
EOF
    # --flash_attn [Flash Attention 2]

if [[ ${1} != "slurm" ]]; then
    export PATH=$HOME/.local/bin/:$PATH
    deepspeed $training_commands
fi


