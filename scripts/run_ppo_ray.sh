DATA=/apdcephfs_sh2/share_300000800/user/antewang/ppo/OpenRLHF-0.5.6/OpenRLHF-0.5.6/custom/dataset/math_level3to5.json
POLICY=/apdcephfs_sh2/share_300000800/user/antewang/ppo/OpenRLHF-0.5.6/Qwen2.5-Math-7B_sft_star/checkpoint-887
OUTPUT=/apdcephfs/share_300000800/user/lfsong/exp.tencent_chat/OpenRLHF/outputs/r1_zero_qwen25_math_7b

set -x 

MEGATRON_REPO=/apdcephfs/share_300000800/user/lfsong/exp.tencent_chat/OpenRLHF/
export PYTHONPATH=${MEGATRON_REPO}:$PYTHONPATH

SEARCH_ALGO="beamsearch"
RM_URL="http://localhost:1234/predict"

   # https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/OpenRLHF/develop-log.md
   # --runtime-env-json='{"working_dir": "./ray_workdir"}' \
   # --load_checkpoint \
   # --apply_chat_template \
ray job submit --address="http://127.0.0.1:8265" \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 4 \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 4 \
   --colocate_actor_ref \
   --reward_num_nodes 0 \
   --reward_num_gpus_per_node 0 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 2 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 2 \
   --pretrain ${POLICY} \
   --remote_rm_url ${RM_URL} \
   --save_path ${OUTPUT} \
   --ckpt_path ${OUTPUT} \
   --micro_train_batch_size 1 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 1 \
   --rollout_batch_size 128 \
   --temperature 0.6 \
   --n_samples_per_prompt 1 \
   --max_samples 100000 \
   --max_epochs 1 \
   --num_episodes 500 \
   --max_len 2048 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.01 \
   --prompt_data ${DATA} \
   --input_key input \
   --normalize_reward \
   --adam_offload \
   --flash_attn \
   --gradient_checkpointing \
   --gradient_checkpointing_use_reentrant \
   --save_steps 25 \
   --search_algo ${SEARCH_ALGO} \
   --search_args "{'search_guidance': 'avg_logp', 'voting': False}"

python3 /jizhi/jizhi2/worker/trainer/occupy_heavy.py
