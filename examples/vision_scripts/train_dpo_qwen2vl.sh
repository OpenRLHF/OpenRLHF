set -x

export HF_DATASETS_OFFLINE=1
wandb online

read -r -d '' training_commands <<EOF
openrlhf.cli.train_vl_dpo \
   --is_visual_model true \
   --task_type dpo \
   --save_path ./checkpoint/qwen2vl_dpo \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps 200 \
   --train_batch_size 8 \
   --micro_train_batch_size 1 \
   --model_arch qwen2_vl \
   --pretrain /home/ckpt/Qwen2-VL-2B-Instruct \
   --bf16 \
   --max_epochs 3 \
   --max_len 4096 \
   --max_samples 1000 \
   --zero_stage 2 \
   --learning_rate 5e-7 \
   --lr_scheduler cosine_with_min_lr \
   --beta 0.1 \
   --dataset dataset/RLHF-V \
   --apply_chat_template \
   --chosen_key chosen \
   --rejected_key rejected \
   --flash_attn \
   --load_checkpoint \
   --gradient_checkpointing

EOF
    # --use_wandb [WANDB_TOKENS] or True (use wandb login command)
    # --ipo [for IPO]
    # --label_smoothing 0.1 [for cDPO]
    # --ref_offload
    # --packing_samples
    # --nll_loss_coef (Regularization with NLL loss)

dataset_setting=" \
   --image_token <|image_pad|> \
   --video_token <|video_pad|> \
   --conversation_tag conversations \
   --role_tag from \
   --content_tag value \
   --original_user_tag human \
   --original_assistant_tag gpt \
   --chosen_key chosen \
   --rejected_key rejected \
"


if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands $dataset_setting
fi
