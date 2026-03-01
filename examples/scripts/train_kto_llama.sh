set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_kto \
   --ckpt_save_path ./checkpoint/llama3-8b-kto \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 64 \
   --micro_train_batch_size 1 \
   --model_name_or_path OpenRLHF/Llama-3-8b-sft-mixture \
   --param_dtype bf16 \
   --max_epochs 1 \
   --max_len 8192 \
   --learning_rate 5e-7 \
   --dataset Dylan2048/ultrafeedback-unpaired-preferences \
   --input_key instruction \
   --output_key response \
   --label_key score \
   --attn_implementation flash_attention_2 \
   --beta 0.1 \
   --max_samples 1024 \
   --gradient_checkpointing
EOF
    # --use_wandb [WANDB_TOKENS] or True (use wandb login command)
    # Resume example (explicit step dir, not /dcp_checkpoint):
    # --resume_from_path /path/to/ckpt/dcp_ckpt/global_step_<N>


if [[ ${1} != "slurm" ]]; then
    torchrun --standalone --nproc-per-node ${NPROC_PER_NODE:-8} -m $training_commands
fi
