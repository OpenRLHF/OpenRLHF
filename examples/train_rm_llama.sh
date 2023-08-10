deepspeed \
     ./scripts/train_rm.py \
     --save_path "./ckpt/7b_llama" \
     --train_batch_size 1 \
     --accumulated_gradient 16 \
     --pretrain "$HOME/scratch/data/llama_hf/7B" \
     --strategy deepspeed \
     --bf16 \
     --max_len 1024 \
     --zero_stage 3\
     --learning_rate 9e-6 \
     --dataset 'Anthropic/hh-rlhf,tasksource/oasst1_pairwise_rlhf_reward' \
     --dataset_probs '0.9,0.1' \
     --load_model './ckpt/7b_llama/sft_model.pt'
