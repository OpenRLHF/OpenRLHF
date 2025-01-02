set -x

export HF_DATASETS_OFFLINE=1

read -r -d '' training_commands <<EOF
openrlhf.cli.train_vl_sft \
   --max_len 2048 \
   --dataset BUAADreamer/llava-en-zh-300k/zh \
   --dataset_config_path examples/vision_scripts/llava_zh_300k.json \
   --train_batch_size 8 \
   --micro_train_batch_size 1 \
   --max_samples 2000 \
   --processing_num_workers 16 \
   --overwrite_cache True \
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

#    --use_wandb True of wandb tokens \
#    --wandb_project xxx \
#    --wandb_run_name xxx

if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi