set -x 


deepspeed ../train_ppo.py \
    --pretrain "/data/model/alpaca-7b" \
    --critic_pretrain "/data/model/alpaca-7b" \
    --reward_pretrain "OpenAssistant/reward-model-deberta-v3-large-v2"\
    --save_path '/data/model/ppo_model_baseline_80K' \
    --micro_train_batch_size 8 \
    --train_batch_size 128 \
    --micro_rollout_batch_size 1 \
    --rollout_batch_size 512 \
    --max_epochs 1 \
    --prompt_max_len 512 \
    --generate_max_len 1024 \
    --zero_stage 2 \
    --bf16 \
    --actor_learning_rate 5e-7 \
    --critic_learning_rate 9e-6 \
    --inference_tp_size 1 \
    --init_kl_coef 0.01 \
    --prompt_data '/data/datasets/finals.json' \
    --prompt_data_probs '1'\
    --normalize_reward \
    --adam_offload \
    --gradient_checkpointing \



