set -x

NUM_GPUS=${1:-8}
# Please make sure install NeMo Gym first and then install OpenRLHF
# pip install git+https://github.com/NVIDIA-NeMo/Gym.git@4616ccd94cc55c98dc233caec83ff4439a3a32b9

ray stop
ray start --head --node-ip-address 0.0.0.0 --num-gpus 8
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/openrlhf"}' \
   -- python3 -u -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node $((NUM_GPUS)) \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node $((NUM_GPUS)) \
   --vllm_num_engines $((NUM_GPUS)) \
   --vllm_tensor_parallel_size 1 \
   --colocate_all_models \
   --agent_func_path /openrlhf/examples/python/agent_func_nemogym_executor.py \
   --pretrain Qwen/Qwen3-1.7B \
   --save_path /openrlhf/examples/test_scripts/checkpoint/Qwen3-1.7B-rlhf-rf_baseline \
   --vllm_gpu_memory_utilization 0.7 \
   --l2 0.01 \
   --micro_train_batch_size 8 \
   --train_batch_size 256 \
   --micro_rollout_batch_size 16 \
   --rollout_batch_size 128  \
   --n_samples_per_prompt 16 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --max_samples 8192 \
   --num_episodes 10 \
   --generate_max_len 4096 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --advantage_estimator reinforce_baseline \
   --init_kl_coef 0 \
   --kl_estimator k2 \
   --prompt_data nvidia/OpenMathInstruct-1 \
   --input_key question \
   --label_key expected_answer \
   --normalize_reward \
   --vllm_enable_sleep \
   --use_dynamic_batch \
   --deepspeed_enable_sleep \
   --gradient_checkpointing \
   --packing_samples \
   --save_steps -1 \
   --enforce_eager  \
   --eps_clip_low_high 0.2 0.28 \
   --enable_prefix_caching \
   --top_p 0.98 \
   --enable_vllm_is_correction 

