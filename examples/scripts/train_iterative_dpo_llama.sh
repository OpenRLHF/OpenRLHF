set -x
export VLLM_WORKER_MULTIPROC_METHOD=spawn

checkSuccess() {
   if [[ $? != 0 ]]; then
      echo "FAILED $1"
      exit 1
   fi
}

mkdir -p ./checkpoint/llama-3-8b-iter-dpo
GENERATE_OUTPUT=./checkpoint/llama-3-8b-iter-dpo/generate.jsonl
RM_OUTPUT=./checkpoint/llama-3-8b-iter-dpo/rm.jsonl
MODEL_OUTPUT_PATH=./checkpoint/llama-3-8b-iter-dpo/checkpoint
ITER_LOG_PATH=null

TRAINING_ITERS=5
ROLLOUT_BATCH_SIZE=10240

POLICY_MODEL_PATH=OpenRLHF/Llama-3-8b-sft-mixture
REF_MODEL_PATH=$POLICY_MODEL_PATH

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
   --max_new_tokens 2048 \
   --prompt_max_len 2048 \
   --dataset OpenRLHF/prompt-collection-v0.1 \
   --input_key context_messages \
   --apply_chat_template \
   --temperature 1.0 \
   --tp_size 4 \
   --best_of_n 8 \
   --enable_prefix_caching \
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
   --post_processor iter_dpo \
   --micro_batch_size 4 \
   --output_path $RM_OUTPUT
EOF
   echo $get_rewards_commands
   deepspeed --module $get_rewards_commands
   checkSuccess "RM"

   read -r -d '' dpo_commands <<EOF
openrlhf.cli.train_dpo \
   --max_len 4096 \
   --dataset $RM_OUTPUT \
   --dataset_probs 1.0 \
   --prompt_key prompt \
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
   deepspeed --module $dpo_commands
   checkSuccess "DPO"

   iter=$((iter + 1))
   if [[ "$ITER_LOG_PATH" != "null" ]]; then
      echo $iter >$ITER_LOG_PATH
   fi
done