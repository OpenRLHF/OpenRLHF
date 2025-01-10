set -x

export HF_DATASETS_OFFLINE=1
wandb online

read -r -d '' training_commands <<EOF
openrlhf.cli.train_vl_sft \
   --is_visual_model true \
   --max_len 2048 \
   --dataset dataset/BUAADreamer/llava-en-zh-300k/zh \
   --train_batch_size 8 \
   --micro_train_batch_size 1 \
   --max_samples 1000 \
   --processing_num_workers 16 \
   --model_arch qwen2_vl \
   --pretrain /home/ckpt/Qwen2-VL-2B-Instruct \
   --save_path ./checkpoint/qwen2vl \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps 50 \
   --zero_stage 2 \
   --max_epochs 3 \
   --bf16 \
   --flash_attn \
   --learning_rate 1e-5 \
   --lr_scheduler constant \
   --load_checkpoint 

EOF
#    --gradient_checkpointing
    # --wandb [WANDB_TOKENS]
    # --packing_samples

dataset_setting=" \
   --image_token <|image_pad|> \
   --video_token <|video_pad|> \
   --conversation_tag messages \
   --role_tag role \
   --content_tag content \
   --original_user_tag user \
   --original_assistant_tag assistant \
"

if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands $dataset_setting
fi