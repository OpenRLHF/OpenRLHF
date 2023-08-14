set -x
export PATH=$HOME/.local/bin/:$PATH
export CUDA_LAUNCH_BLOCKING=1

deepspeed ../train_sft.py \
    --max_len 2048 \
    --dataset 'yahma/alpaca-cleaned' \
    --train_batch_size 128 \
    --micro_train_batch_size 1 \
    --pretrain "meta-llama/Llama-2-7b-hf" \
    --save_path "./ckpt/7b_llama" \
    --zero_stage 2 \
    --max_epochs 2 \
    --bf16 \
    --learning_rate 5e-6 \
    # --gradient_checkpointing
