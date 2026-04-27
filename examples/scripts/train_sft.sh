set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
   --data.max_len 2048 \
   --data.dataset Open-Orca/OpenOrca \
   --data.input_key question \
   --data.output_key response \
   --train.batch_size 256 \
   --train.micro_batch_size 2 \
   --data.max_samples 500000 \
   --model.model_name_or_path meta-llama/Meta-Llama-3-8B \
   --ckpt.output_dir ./checkpoint/llama3-8b-sft \
   --ckpt.save_steps -1 \
   --logger.logging_steps 1 \
   --eval.steps -1 \
   --train.max_epochs 1 \
   --fsdp.param_dtype bf16 \
   --fsdp.attn_implementation flash_attention_2 \
   --adam.lr 5e-6 \
   --ckpt.load_enable \
   --model.gradient_checkpointing_enable
EOF
    # --wandb [WANDB_TOKENS]

if [[ ${1} != "slurm" ]]; then
    torchrun --standalone --nproc_per_node ${NPROC_PER_NODE:-8} -m $training_commands
fi
