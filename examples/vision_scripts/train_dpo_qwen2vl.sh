set -x

export HF_DATASETS_OFFLINE=1

read -r -d '' training_commands <<EOF
openrlhf.cli.train_vl_dpo \
   --task_type dpo \
   --save_path ./checkpoint/qwen2vl_dpo \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps 50 \
   --train_batch_size 8 \
   --micro_train_batch_size 1 \
   --model_arch qwen2_vl \
   --pretrain /home/ckpt/Qwen2-VL-2B-Instruct \
   --bf16 \
   --max_epochs 1 \
   --max_len 4096 \
   --max_samples 2000 \
   --zero_stage 2 \
   --learning_rate 1e-6 \
   --lr_scheduler constant \
   --beta 0.1 \
   --dataset dataset/RLHF-V \
   --dataset_config_path examples/vision_scripts/rlhf_v.json \
   --apply_chat_template \
   --chosen_key chosen \
   --rejected_key rejected \
   --flash_attn \
   --load_checkpoint \
   --gradient_checkpointing
EOF
    # --use_wandb True of wandb tokens \
    # --wandb_project xxx \
    # --wandb_run_name xxx
    # --ipo [for IPO]
    # --label_smoothing 0.1 [for cDPO]
    # --ref_offload
    # --packing_samples
    # --nll_loss_coef (Regularization with NLL loss)


if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi
