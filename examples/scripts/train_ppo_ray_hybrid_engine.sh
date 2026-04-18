set -x

python3 -m openrlhf.cli.train_ppo_ray \
   --ref.num_nodes 1 \
   --ref.num_gpus_per_node 8 \
   --reward.num_nodes 1 \
   --reward.num_gpus_per_node 8 \
   --critic.num_nodes 1 \
   --critic.num_gpus_per_node 8 \
   --actor.num_nodes 1 \
   --actor.num_gpus_per_node 8 \
   --vllm_num_engines 4 \
   --vllm_tensor_parallel_size 2 \
   --colocate_all_models \
   --vllm_gpu_memory_utilization 0.5 \
   --actor.model_name_or_path OpenRLHF/Llama-3-8b-sft-mixture \
   --reward.model_name_or_path OpenRLHF/Llama-3-8b-rm-700k \
   --save_path /openrlhf/examples/test_scripts/final/llama3-8b-rlhf \
   --ckpt_path /openrlhf/examples/test_scripts/ckpt/llama3-8b-rlhf \
   --save_hf_ckpt \
   --train_batch_size 128 \
   --rollout_batch_size 1024 \
   --n_samples_per_prompt 1 \
   --max_epochs 1 \
   --max_len 2048 \
   --max_samples 100000 \
   --zero_stage 3 \
   --param_dtype bf16 \
   --actor.adam.lr 5e-7 \
   --critic.adam.lr 9e-6 \
   --init_kl_coef 0.01 \
   --prompt_data OpenRLHF/prompt-collection-v0.1 \
   --input_key context_messages \
   --apply_chat_template \
   --reward.normalize \
   --actor.gradient_checkpointing \
   --packing_samples \
   --vllm_sync_backend nccl \
   --enforce_eager \
   --vllm_enable_sleep \
   --deepspeed_enable_sleep \
   --use_dynamic_batch \
   --train_max_tokens_per_gpu 16384 \
   --enable_vllm_is_correction

# Enable tensor parallelism for DeepSpeed
#    --ds_tensor_parallel_size 2 \

# Enable Ring-Attention
#    --actor.ring_attn_size 4 \
#    --actor.ring_attn_head_stride 2 \