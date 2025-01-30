POLICY=/apdcephfs_sh2/share_300000800/user/antewang/ppo/OpenRLHF-0.5.6/Qwen2.5-Math-1.5B_sft_star
CRITIC=/apdcephfs_sh2/share_300000800/user/antewang/ppo/OpenRLHF-0.5.6/Qwen2.5-Math-1.5B_sft_value
OUTPUT=/apdcephfs/share_300000800/user/lfsong/exp.tencent_chat/output/Qwen2.5-Math-1.5B_gsm8k_math_ppo_vote_fp32

SCRIPT_DIR="/apdcephfs/share_300000800/user/lfsong/exp.tencent_chat/hunyuan-prover/mcts_search/train"

deepspeed ${SCRIPT_DIR}/train_ppo.py \
  --pretrain ${POLICY} \
  --critic_pretrain ${CRITIC} \
  --remote_rm_url http://localhost:1234/predict \
  --save_path ${OUTPUT} \
  --save_steps -1 \
  --logging_steps 1 \
  --eval_steps -1 \
  --micro_train_batch_size 2 \
  --train_batch_size 256 \
  --micro_rollout_batch_size 2 \
  --rollout_batch_size 256 \
  --max_epochs 1 \
  --num_episodes 4 \
  --prompt_max_len 512 \
  --generate_max_len 1024 \
  --zero_stage 2 \
  --actor_learning_rate 5e-7 \
  --critic_learning_rate 1e-6 \
  --init_kl_coef 0.01 \
  --prompt_data dataset/train_gsm8k_math.jsonl \
  --input_key input \
  --max_samples 100000 \
  --normalize_reward \
  --adam_offload \
  --gradient_checkpointing \
  --search_algo bestofn

python3 occupy_heavy.py

# --bf16 \
# --grad_accum_dtype fp32 \