set -x
export VLLM_WORKER_MULTIPROC_METHOD=spawn

checkSuccess() {
   if [[ $? != 0 ]]; then
      echo "FAILED $1"
      exit 1
   fi
}

mkdir -p ./checkpoint/llama-3-8b-rejection
GENERATE_OUTPUT=./checkpoint/llama-3-8b-rejection/generate.jsonl
RM_OUTPUT=./checkpoint/llama-3-8b-rejection/rm.jsonl
ITER_LOG_PATH=./checkpoint/llama-3-8b-rejection/iter.log
MODEL_OUTPUT_PATH=./checkpoint/llama-3-8b-rejection

TRAINING_ITERS=10
ROLLOUT_BATCH_SIZE=10240

POLICY_MODEL_PATH=OpenRLHF/Llama-3-8b-sft-mixture

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
openrlhf.cli.batch_inference
   --eval_task generate_vllm \
   --pretrain $POLICY_MODEL_PATH \
   --bf16 \
   --max_new_tokens 2048 \
   --prompt_max_len 2048 \
   --dataset OpenRLHF/prompt-collection-v0.1 \
   --input_key context_messages \
   --apply_chat_template \
   --temperature 0.9
   --zero_stage 0 \
   --best_of_n 4 \
   --enable_prefix_caching \
   --tp_size 4 \
   --max_num_seqs 64 \
   --iter $iter \
   --rollout_batch_size $ROLLOUT_BATCH_SIZE \
   --output_path $GENERATE_OUTPUT
EOF
   echo $generate_commands
   python -m $generate_commands
   checkSuccess "GENERATE"

   read -r -d '' get_rewards_commands <<EOF
openrlhf.cli.batch_inference
   --eval_task rm \
   --pretrain OpenRLHF/Llama-3-8b-rm-mixture \
   --bf16 \
   --max_len 4096 \
   --dataset $GENERATE_OUTPUT  \
   --dataset_probs 1.0 \
   --zero_stage 0 \
   --post_processor rs \
   --micro_batch_size 4 \
   --output_path $RM_OUTPUT
EOF
   echo $get_rewards_commands
   deepspeed --module $get_rewards_commands
   checkSuccess "RM"

   read -r -d '' sft_commands <<EOF
openrlhf.cli.train_sft \
   --max_len 4096 \
   --dataset $RM_OUTPUT \
   --dataset_probs 1.0 \
   --train_batch_size 128 \
   --micro_train_batch_size 2 \
   --pretrain $POLICY_MODEL_PATH \
   --save_path ./checkpoint/llama-3-8b-rejection \
   --input_template "" \
   --zero_stage 2 \
   --max_epochs 1 \
   --bf16 \
   --learning_rate 2e-6 \
   --gradient_checkpointing
EOF
   echo $sft_commands
   deepspeed --module $sft_commands
   checkSuccess "SFT"

   iter=$((iter + 1))
   if [[ "$ITER_LOG_PATH" != "null" ]]; then
      echo $iter >$ITER_LOG_PATH
   fi
done