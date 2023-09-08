set -x 
export PATH=$HOME/.local/bin/:$PATH

CUDA_VISIBLE_DEVICES=0 \
deepspeed ../inference.py \
    --eval_task "rm" \
    --pretrain "/mnt/workspace/lvyang/model_ckpts/models--meta-llama--Llama-2-7b-chat-hf/snapshots/08751db2aca9bf2f7f80d2e516117a53d7450235" \
    --bf16 \
    --load_model "/mnt/data/project/lvyang_nas/lvyang/model_ckpts/7b_llama_chat_rm/rm_model.pt"