set -x
export PATH=$HOME/.local/bin/:$PATH

deepspeed \
     ../train_sft.py \
     --max_len 2048 \
    --dataset 'yahma/alpaca-cleaned' \
    --train_batch_size 1 \
    --accumulated_gradient 16 \
    --pretrain "$HOME/scratch/data/llama_hf/7B" \
    --save_path "./ckpt/7b_llama" \
    --zero_stage 2 \
    --max_epochs 1 \
    --bf16 \
    --learning_rate 9e-6 \
    --gradient_checkpointing
