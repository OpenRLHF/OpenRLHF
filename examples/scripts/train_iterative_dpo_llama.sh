set -x

mkdir -p ./ckpt/7b_llama_iter_dpo
GENERATE_OUTPUT=./ckpt/7b_llama_iter_dpo/generate.jsonl
RM_OUTPUT=./ckpt/7b_llama_iter_dpo/rm.jsonl
MODEL_OUTPUT_PATH=./ckpt/7b_llama_iter_dpo/ckpt
ITER_LOG_PATH=null

TRAINING_ITERS=5
ROLLOUT_BATCH_SIZE=10240

POLICY_MODEL_PATH=OpenLLMAI/Llama-2-7b-sft-model-ocra-500k
REF_MODEL_PATH=$POLICY_MODEL_PATH

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
fi

while (($iter < $TRAINING_ITERS)); do
    echo "Iter: $iter"
    # Use latest model if past first iteration
    if ((iter > 0)); then
        POLICY_MODEL_PATH=$MODEL_OUTPUT_PATH
    fi

    read -r -d '' generate_commands <<EOF
../batch_inference.py
    --eval_task generate_vllm \
    --pretrain $POLICY_MODEL_PATH \
    --max_new_tokens 1024 \
    --dataset Open-Orca/OpenOrca,Dahoas/full-hh-rlhf  \
    --dataset_probs 0.5,0.5 \
    --temperature 1.0 \
    --tp_size 4 \
    --best_of_n 16 \
    --max_num_seqs 128 \
    --iter $iter \
    --rollout_batch_size $ROLLOUT_BATCH_SIZE \
    --output_path $GENERATE_OUTPUT
EOF
    echo $generate_commands
    python $generate_commands
    checkSuccess "GENERATE"

    read -r -d '' get_rewards_commands <<EOF
../batch_inference.py
    --eval_task rm \
    --pretrain OpenLLMAI/Llama-2-7b-rm-anthropic_hh-lmsys-oasst-webgpt \
    --bf16 \
    --max_len 2048 \
    --dataset $GENERATE_OUTPUT  \
    --dataset_probs 1.0 \
    --zero_stage 0 \
    --post_processor iter_dpo \
    --micro_batch_size 4 \
    --output_path $RM_OUTPUT
EOF
    echo $get_rewards_commands
    deepspeed $get_rewards_commands
    checkSuccess "RM"

    read -r -d '' dpo_commands <<EOF
../train_dpo.py \
    --max_len 2048 \
    --dataset $RM_OUTPUT \
    --dataset_probs 1.0 \
    --train_batch_size 128 \
    --micro_train_batch_size 2 \
    --pretrain $POLICY_MODEL_PATH \
    --ref_pretrain $REF_MODEL_PATH \
    --save_path $MODEL_OUTPUT_PATH \
    --zero_stage 3 \
    --max_epochs 1 \
    --bf16 \
    --learning_rate 5e-7 \
    --gradient_checkpointing
EOF
    echo $dpo_commands
    deepspeed $dpo_commands
    checkSuccess "DPO"

    iter=$((iter + 1))
    if [[ "$ITER_LOG_PATH" != "null" ]]; then
        echo $iter >$ITER_LOG_PATH
    fi
done
