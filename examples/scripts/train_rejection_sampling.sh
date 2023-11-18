set -x

mkdir -p ./ckpt/7b_llama_rs
GENERATE_OUTPUT=./ckpt/7b_llama_rs/generate.jsonl
RM_OUTPUT=./ckpt/7b_llama_rs/rm.jsonl
MODEL_OUTPUT_PATH=./ckpt/7b_llama_rs/sft_model.pt
ITER_LOG_PATH=null

TRAINING_ITERS=20
ROLLOUT_BATCH_SIZE=2048

POLICY_MODEL_PATH=./ckpt/7b_llama/sft_model_ocra.pt
REWARD_MODEL_PATH=./ckpt/7b_llama/rm_model_anthropic_oasst_lmsys_webgpt.pt

checkSuccess() {
    if [[ $? != 0 ]]; then
        echo "FAILED $1"
        exit 1
    fi
}

export PATH=$HOME/.local/bin/:$PATH

iter=0
if [ -f $ITER_LOG_PATH ]; then
    iter=$(cat $ITER_LOG_PATH)
    echo "Read iters: $iter" &>>$LOGS_PATH
fi

while (($iter < $TRAINING_ITERS)); do
    echo "Iter: $iter"
    # Use latest model if past first iteration
    if ((iter > 0)); then
        POLICY_MODEL_PATH=$MODEL_OUTPUT_PATH
    fi

    read -r -d '' generate_commands <<EOF
../batch_inference.py
    --eval_task generate \
    --pretrain meta-llama/Llama-2-7b-hf \
    --bf16 \
    --load_model $POLICY_MODEL_PATH \
    --max_len 2048 \
    --dataset Open-Orca/OpenOrca,Dahoas/full-hh-rlhf  \
    --dataset_probs 0.5,0.5 \
    --temperature 0.9
    --zero_stage 0 \
    --best_of_n 16 \
    --micro_batch_size 16 \
    --iter $iter \
    --rollout_batch_size $ROLLOUT_BATCH_SIZE \
    --output_path $GENERATE_OUTPUT
EOF
    echo $generate_commands
    deepspeed $generate_commands
    checkSuccess "GENERATE"

    read -r -d '' get_rewards_commands <<EOF
../batch_inference.py
    --eval_task rm \
    --pretrain meta-llama/Llama-2-7b-hf \
    --bf16 \
    --load_model $REWARD_MODEL_PATH \
    --max_len 2048 \
    --dataset $GENERATE_OUTPUT  \
    --dataset_probs 1.0 \
    --zero_stage 0 \
    --post_processor rs \
    --micro_batch_size 4 \
    --output_path $RM_OUTPUT
EOF
    echo $get_rewards_commands
    deepspeed $get_rewards_commands
    checkSuccess "RM"

    read -r -d '' sft_commands <<EOF
../train_sft.py \
    --max_len 2048 \
    --dataset $RM_OUTPUT \
    --dataset_probs 1.0 \
    --train_batch_size 128 \
    --micro_train_batch_size 2 \
    --pretrain meta-llama/Llama-2-7b-hf \
    --save_path ./ckpt/7b_llama_rs \
    --load_model $POLICY_MODEL_PATH \
    --lr_scheduler constant \
    --zero_stage 2 \
    --max_epochs 1 \
    --bf16 \
    --learning_rate 2e-6 \
    --gradient_checkpointing
EOF
    echo $sft_commands
    deepspeed $sft_commands
    checkSuccess "SFT"

    iter=$((iter + 1))
    if [[ "$ITER_LOG_PATH" != "null" ]]; then
        echo $iter >$ITER_LOG_PATH
    fi
done
