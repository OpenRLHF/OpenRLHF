set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
   --max_len 2048 \
   --dataset Open-Orca/OpenOrca \
   --input_key question \
   --output_key response \
   --train_batch_size 256 \
   --micro_train_batch_size 2 \
   --max_samples 500000 \
   --model_name_or_path meta-llama/Meta-Llama-3-8B \
   --ckpt_save_path ./checkpoint/llama3-8b-sft \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --max_epochs 1 \
   --param_dtype bf16 \
   --attn_implementation flash_attention_2 \
   --learning_rate 5e-6 \
   --packing_samples \
   --gradient_checkpointing
EOF
    # --wandb [WANDB_TOKENS]
    # --packing_samples
    # Resume example (explicit step dir, not /dcp_checkpoint):
    # --resume_from_path /path/to/ckpt/dcp_ckpt/global_step_<N>

if [[ ${1} != "slurm" ]]; then
    torchrun --standalone --nproc-per-node ${NPROC_PER_NODE:-8} -m $training_commands
fi
