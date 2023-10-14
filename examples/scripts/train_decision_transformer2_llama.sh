set -x

mkdir -p ./ckpt/7b_llama_dt2
RM_OUTPUT=./ckpt/7b_llama_dt2/rm.jsonl

# Decision Transformer Alignment without Generation
# Faster than train_decision_transformer_llama.sh

read -r -d '' get_rewards_commands <<EOF
../batch_inference.py
    --eval_task rm \
    --pretrain meta-llama/Llama-2-7b-hf \
    --bf16 \
    --load_model ./ckpt/7b_llama/rm_model_anthropic_oasst_lmsys_webgpt.pt
    --max_len 2048 \
    --dataset Open-Orca/OpenOrca,Dahoas/full-hh-rlhf,tasksource/oasst1_pairwise_rlhf_reward \
    --dataset_probs 0.2,0.7,0.1 \
    --max_samples 180000 \
    --zero_stage 0 \
    --post_processor dt \
    --micro_batch_size 4 \
    --output_path $RM_OUTPUT
    --normalize_reward
EOF

read -r -d '' sft_commands <<EOF
../train_sft.py \
    --max_len 2048 \
    --dataset $RM_OUTPUT \
    --dataset_probs 1.0 \
    --train_batch_size 128 \
    --micro_train_batch_size 2 \
    --pretrain meta-llama/Llama-2-7b-hf \
    --save_path ./ckpt/7b_llama_dt2 \
    --zero_stage 2 \
    --max_epochs 1 \
    --bf16 \
    --learning_rate 5e-6 \
    --gradient_checkpointing \
    --save_hf_model
EOF


checkSuccess() {
    if [[ $? != 0 ]]; then
        echo "FAILED $1"
        exit 1
    fi
}

if [[ ${1} != "slurm" ]]; then
    export PATH=$HOME/.local/bin/:$PATH

    if [ ! -e $RM_OUTPUT ]; then
        deepspeed $get_rewards_commands
        checkSuccess "RM"
    fi
    deepspeed $sft_commands
fi