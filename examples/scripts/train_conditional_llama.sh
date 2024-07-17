set -x

checkSuccess() {
    if [[ $? != 0 ]]; then
        echo "FAILED $1"
        exit 1
    fi
}

mkdir -p ./checkpoint/llama-2-8b-csft
RM_OUTPUT=./checkpoint/llama-2-8b-csft/rm.jsonl

read -r -d '' get_rewards_commands <<EOF
openrlhf.cli.batch_inference \
    --eval_task rm \
    --pretrain OpenRLHF/Llama-3-8b-rm-mixture \
    --bf16 \
    --max_len 4096 \
    --dataset OpenRLHF/preference_dataset_mixture2_and_safe_pku \
    --input_key chosen \
    --apply_chat_template \
    --max_samples 128000 \
    --zero_stage 0 \
    --post_processor csft \
    --normalize_reward
    --micro_batch_size 4 \
    --output_path $RM_OUTPUT
EOF

read -r -d '' sft_commands <<EOF
openrlhf.cli.train_sft \
    --max_len 4096 \
    --dataset $RM_OUTPUT \
    --dataset_probs 1.0 \
    --train_batch_size 128 \
    --micro_train_batch_size 2 \
    --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
    --save_path ./checkpoint/llama-3-8b-csft \
    --zero_stage 2 \
    --max_epochs 1 \
    --bf16 \
    --learning_rate 5e-6 \
    --gradient_checkpointing
EOF

if [ ! -e $RM_OUTPUT ]; then
    deepspeed --module $get_rewards_commands
    checkSuccess "RM"
fi
deepspeed --module $sft_commands