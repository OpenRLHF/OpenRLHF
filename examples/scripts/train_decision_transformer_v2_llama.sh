set -x

mkdir -p ./ckpt/7b_llama_dt
RM_OUTPUT=./ckpt/7b_llama_dt/rm.jsonl

# We found that the DT without generation achieved similar performance to the DT with generation version
# And DT without generation is significantly faster than DT with generation.
#
# Alpaca_eval results
#
# DT (8 hours with 4 A100) vs OCRA SFT
#                win_rate  standard_error  n_total  avg_length
# Current model     61.64            3.87      159        1956
#
# DT - non generation (4 hours with 4 A100) vs OCRA SFT
#                win_rate  standard_error  n_total  avg_length
# Current model     61.95            3.85      159        1568

read -r -d '' get_rewards_commands <<EOF
../batch_inference.py
    --eval_task rm \
    --pretrain meta-llama/Llama-2-7b-hf \
    --bf16 \
    --load_model ./ckpt/7b_llama/rm_model_anthropic_oasst_lmsys_webgpt.pt
    --max_len 2048 \
    --dataset Open-Orca/OpenOrca,Dahoas/full-hh-rlhf \
    --dataset_probs 0.5,0.5 \
    --max_samples 128000 \
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
    --save_path ./ckpt/7b_llama_dt \
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