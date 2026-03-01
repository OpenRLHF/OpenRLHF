set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_prm \
   --ckpt_save_path ./checkpoint/mistal-7b-prm \
   --save_steps 500 \
   --logging_steps 1 \
   --eval_steps 100 \
   --train_batch_size 256 \
   --micro_train_batch_size 8 \
   --model_name_or_path mistralai/Mistral-7B-v0.1  \
   --param_dtype bf16 \
   --max_epochs 1 \
   --max_len 8192 \
   --learning_rate 1e-6 \
   --dataset zhuzilin/Math-Shepherd \
   --input_key input \
   --label_key value \
   --attn_implementation flash_attention_2 \
   --gradient_checkpointing \
   --packing_samples \
   --wandb_group prm \
   --placeholder_token ки \
   --reward_tokens + -
EOF
     # --use_wandb [WANDB_TOKENS] or True (use wandb login command)
     # --packing_samples
     # Resume example (explicit step dir, not /dcp_checkpoint):
     # --resume_from_path /path/to/ckpt/dcp_ckpt/global_step_<N>


if [[ ${1} != "slurm" ]]; then
    torchrun --standalone --nproc-per-node ${NPROC_PER_NODE:-8} -m $training_commands
fi
