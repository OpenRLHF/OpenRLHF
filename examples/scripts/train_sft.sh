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
   --ds.zero_stage 2 \
   --train.max_epochs 1 \
   --ds.param_dtype bf16 \
   --ds.attn_implementation flash_attention_2 \
   --adam.lr 5e-6 \
   --ckpt.load_enable \
   --ds.packing_samples \
   --model.gradient_checkpointing_enable
EOF
    # --wandb [WANDB_TOKENS]
    # --ds.packing_samples

if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi