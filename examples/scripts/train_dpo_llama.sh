set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_dpo \
   --ckpt.output_dir ./checkpoint/llama3-8b-dpo \
   --ckpt.save_steps -1 \
   --logger.logging_steps 1 \
   --eval.steps -1 \
   --train.batch_size 256 \
   --train.micro_batch_size 1 \
   --model.model_name_or_path OpenRLHF/Llama-3-8b-sft-mixture \
   --fsdp.param_dtype bf16 \
   --train.max_epochs 1 \
   --data.max_len 8192 \
   --adam.lr 5e-7 \
   --model.beta 0.1 \
   --data.dataset OpenRLHF/preference_dataset_mixture2_and_safe_pku \
   --data.apply_chat_template \
   --data.chosen_key chosen \
   --data.rejected_key rejected \
   --fsdp.attn_implementation flash_attention_2 \
   --ckpt.load_enable \
   --fsdp.packing_samples \
   --model.gradient_checkpointing_enable
EOF
    # --logger.wandb.key [WANDB_TOKENS] or True (use wandb login command)
    # --model.ipo_enable [for IPO]
    # --model.label_smoothing 0.1 [for cDPO]
    # --ref.offload
    # --fsdp.packing_samples
    # --model.nll_loss_coef (Regularization with NLL loss)


if [[ ${1} != "slurm" ]]; then
    torchrun --standalone --nproc_per_node ${NPROC_PER_NODE:-8} -m $training_commands
fi
