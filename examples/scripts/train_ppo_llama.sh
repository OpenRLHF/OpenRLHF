set -x 

deepspeed \
    ../train_ppo.py --pretrain "$HOME/scratch/data/llama_hf/7B" \
    --strategy deepspeed\
    --critic_pretrain "$HOME/scratch/data/llama_hf/7B" \
    --reward_model_path "./ckpt/7b_llama/rm_model.pt" \
    --sft_model_path './ckpt/7b_llama/sft_model.pt' \
    --save_path './ckpt/7b_llama' \
    --train_batch_size 1 \
    --accumulated_gradient 16 \
    --experience_batch_size 1 \
    --update_timesteps 128 \
    --max_epochs 1\
    --prompt_max_len 1024 \
    --generate_max_len 1024 \
    --zero_stage 2 \
    --bf16\
    --actor_learning_rate 5e-7\
    --critic_learning_rate 9e-6\
    --inference_tp_size 4 \
    --init_kl_coef 0.01\
    --prompt_data "Dahoas/full-hh-rlhf,tasksource/oasst1_pairwise_rlhf_reward" \
    --prompt_data_probs "0.8,0.2" \
    --normalize_reward \
    --gradient_checkpointing
    # --pretrain_data "EleutherAI/pile"
    # --kl_target 0.8\

