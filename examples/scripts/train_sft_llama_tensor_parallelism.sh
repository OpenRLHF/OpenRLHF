set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
   --max_len 2048 \
   --dataset Open-Orca/OpenOrca \
   --input_key question \
   --output_key response \
   --train_batch_size 256 \
   --micro_train_batch_size 4 \
   --max_samples 500000 \
   --pretrain meta-llama/Meta-Llama-3-8B \
   --save_path ./checkpoint/llama3-8b-sft \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 2 \
   --max_epochs 1 \
   --bf16 \
   --attn_implementation flash_attention_2 \
   --learning_rate 5e-6 \
   --load_checkpoint \
   --gradient_checkpointing \
   --packing_samples \
   --ds_tensor_parallel_size 2
EOF
    # --wandb [WANDB_TOKENS]
    # --packing_samples

if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi