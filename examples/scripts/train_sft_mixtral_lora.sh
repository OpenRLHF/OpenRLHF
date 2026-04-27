set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
    --data.max_len 2048 \
    --data.dataset Open-Orca/OpenOrca \
    --data.input_key question \
    --data.output_key response \
    --train.batch_size 128 \
    --train.micro_batch_size 4 \
    --data.max_samples 500000 \
    --model.model_name_or_path mistralai/Mixtral-8x7B-v0.1 \
    --ckpt.output_dir ./checkpoint/mixtral-sft-lora \
    --ckpt.save_steps -1 \
    --logger.logging_steps 1 \
    --eval.steps -1 \
    --train.max_epochs 1 \
    --fsdp.param_dtype bf16 \
    --model.gradient_checkpointing_enable \
    --fsdp.attn_implementation flash_attention_2 \
    --adam.lr 5e-6 \
    --fsdp.lora.rank 64 \
    --fsdp.lora.alpha 64 \
    --fsdp.packing_samples \
    --model.aux_loss_coef 0.001
EOF

if [[ ${1} != "slurm" ]]; then
    torchrun --standalone --nproc_per_node ${NPROC_PER_NODE:-8} -m $training_commands
fi
