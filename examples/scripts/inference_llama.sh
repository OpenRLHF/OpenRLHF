set -x 
export PATH=$HOME/.local/bin/:$PATH

CUDA_VISIBLE_DEVICES=0 \
deepspeed ../inference.py \
    --eval_task "generate" \
    --pretrain "meta-llama/Llama-2-7b-chat-hf" \
    --bf16 \
    --load_model $1