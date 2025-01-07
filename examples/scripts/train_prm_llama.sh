set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_prm \
   --save_path ./checkpoint/llama3-8b-prm \
   --save_steps 500 \
   --logging_steps 1 \
   --eval_steps 100 \
   --train_batch_size 256 \
   --micro_train_batch_size 8 \
   --pretrain meta-llama/Llama-3-8B  \
   --bf16 \
   --max_epochs 1 \
   --max_len 8192 \
   --zero_stage 3 \
   --learning_rate 1e-6 \
   --dataset zhuzilin/Math-Shepherd \
   --input_key input \
   --label_key value \
   --flash_attn \
   --load_checkpoint \
   --gradient_checkpointing \
   --packing_samples \
   --wandb_group prm \
   --placeholder_token ки \
   --reward_tokens + - \
   --specialize_reward_tokens \
   --placeholder_token_in_tokenizer "<|reserved_special_token_0|>"
EOF
     # --use_wandb [WANDB_TOKENS] or True (use wandb login command)
     # --packing_samples


if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi
