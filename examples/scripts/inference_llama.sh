set -x 
export PATH=$HOME/.local/bin/:$PATH

CUDA_VISIBLE_DEVICES=0 \
deepspeed ../inference.py \
    --pretrain "meta-llama/Llama-2-7b-chat-hf" \
    --bf16 \
    --input "$2" \
    --load_model "$1"