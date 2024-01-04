set -x 
export PATH=$HOME/.local/bin/:$PATH

CUDA_VISIBLE_DEVICES=0 \
deepspeed ../interactive_chat.py \
    --pretrain $1 \
    --bf16