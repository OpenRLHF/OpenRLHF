set -x 
export PATH=$HOME/.local/bin/:$PATH

CUDA_VISIBLE_DEVICES=0 \
deepspeed ../inference.py \
    --eval_task "rm" \
    --pretrain "meta-llama/Llama-2-7b-chat-hf" \
    --bf16 \
    --load_model "./ckpt/7b_llama/rm_model_anthropic_oasst_lmsys_webgpt.pt"