set -x 

CUDA_VISIBLE_DEVICES=0 deepspeed scripts/inference.py \
    --pretrain "$HOME/scratch/data/llama_hf/7B" \
    --strategy deepspeed \
    --bf16 \
    --load_model "./ckpt/7b_llama_1024/sft_model.pt"\
    --input "$1"