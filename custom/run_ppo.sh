POLICY=/apdcephfs_cq11/share_1603164/data/antewang/trained_models/Qwen-2.5-Math-1.5B_sft_star_fp32/checkpoint-1738
CRITIC=/apdcephfs_cq11/share_1603164/data/antewang/trained_models/Qwen2.5-Math-1.5B_sft_value_ep1_bsz64_lr5e-6_fp32/checkpoint-1489

# # launch the master node of ray in container
# ray start --head --node-ip-address 0.0.0.0 --num-gpus 8

# ray job submit --address="http://127.0.0.1:8265" \
#    --runtime-env-json='{"working_dir": "./logs"}' \
#    -- python3 -m openrlhf.cli.train_ppo_ray \
#    --ref_num_nodes 1 \
#    --ref_num_gpus_per_node 2 \
#    --reward_num_nodes 1 \
#    --reward_num_gpus_per_node 2 \
#    --critic_num_nodes 1 \
#    --critic_num_gpus_per_node 2 \
#    --actor_num_nodes 1 \
#    --actor_num_gpus_per_node 2 \
#    --vllm_num_engines 2 \
#    --vllm_tensor_parallel_size 2 \
#    --colocate_critic_reward \
#    --colocate_actor_ref \
#    --pretrain ${POLICY} \
#    --critic_pretrain ${CRITIC} \
#    --remote_rm_url http://localhost:1234/predict \
#    --save_path ./checkpoint/Qwen2.5-Math-1.5B_gsm8k_math_ppo \
#    --micro_train_batch_size 8 \
#    --train_batch_size 128 \
#    --micro_rollout_batch_size 32 \
#    --rollout_batch_size 1024 \
#    --max_samples 100000 \
#    --max_epochs 1 \
#    --prompt_max_len 1024 \
#    --generate_max_len 1024 \
#    --zero_stage 3 \
#    --actor_learning_rate 5e-7 \
#    --critic_learning_rate 9e-6 \
#    --init_kl_coef 0.01 \
#    --prompt_data dataset/train_gsm8k_math.jsonl \
#    --input_key input \
#    --normalize_reward \
#    --packing_samples \
#    --adam_offload \
#    --flash_attn \
#    --gradient_checkpointing \
#    --load_checkpoint

deepspeed --module train_ppo \
  --pretrain ${POLICY} \
  --critic_pretrain ${CRITIC} \
  --remote_rm_url http://localhost:1234/predict \
  --save_path ./checkpoint/Qwen2.5-Math-1.5B_gsm8k_math_ppo_beamsearch \
  --save_steps 58 \
  --logging_steps 1 \
  --eval_steps -1 \
  --micro_train_batch_size 2 \
  --train_batch_size 64 \
  --micro_rollout_batch_size 2 \
  --rollout_batch_size 256 \
  --max_epochs 1 \
  --num_episodes 4 \
  --prompt_max_len 512 \
  --generate_max_len 768 \
  --zero_stage 2 \
  --actor_learning_rate 5e-7 \
  --critic_learning_rate 1e-6 \
  --init_kl_coef 0.01 \
  --prompt_data dataset/train_gsm8k_math.jsonl \
  --input_key input \
  --max_samples 100000 \
  --normalize_reward \
  --adam_offload \
  --gradient_checkpointing

python3 /apdcephfs/private_antewang/self-endorsement/occupy_heavy.py
