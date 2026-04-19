set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_rm \
   --ckpt.output_dir ./checkpoint/llama3-8b-rm \
   --ckpt.save_steps -1 \
   --train.logging_steps 1 \
   --eval.steps -1 \
   --train.batch_size 256 \
   --train.micro_batch_size 1 \
   --actor.model_name_or_path OpenRLHF/Llama-3-8b-sft-mixture \
   --ds.param_dtype bf16 \
   --train.max_epochs 1 \
   --data.max_len 8192 \
   --ds.zero_stage 3 \
   --adam.lr 9e-6 \
   --data.dataset OpenRLHF/preference_dataset_mixture2_and_safe_pku \
   --data.apply_chat_template \
   --chosen_key chosen \
   --rejected_key rejected \
   --actor.attn_implementation flash_attention_2 \
   --ckpt.load \
   --data.packing_samples \
   --actor.gradient_checkpointing
EOF
     # --logger.wandb.key [WANDB_TOKENS] or True (use wandb login command)
     # --data.packing_samples


if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi
