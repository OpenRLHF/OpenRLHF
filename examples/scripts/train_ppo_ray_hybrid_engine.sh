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
   --vllm.num_engines 4 \
   --vllm.tensor_parallel_size 2 \
   --train.colocate_all \
   --vllm.gpu_memory_utilization 0.5 \
   --actor.model_name_or_path OpenRLHF/Llama-3-8b-sft-mixture \
   --reward.model_name_or_path OpenRLHF/Llama-3-8b-rm-700k \
   --ckpt.output_dir /openrlhf/examples/test_scripts/final/llama3-8b-rlhf \
   --ckpt.path /openrlhf/examples/test_scripts/ckpt/llama3-8b-rlhf \
   --ckpt.save_hf \
   --train.batch_size 128 \
   --rollout.batch_size 1024 \
   --rollout.n_samples_per_prompt 1 \
   --train.max_epochs 1 \
   --data.max_len 2048 \
   --data.max_samples 100000 \
   --ds.zero_stage 3 \
   --ds.param_dtype bf16 \
   --actor.adam.lr 5e-7 \
   --critic.adam.lr 9e-6 \
   --algo.kl.init_coef 0.01 \
   --data.prompt_dataset OpenRLHF/prompt-collection-v0.1 \
   --data.input_key context_messages \
   --data.apply_chat_template \
   --reward.normalize_enable \
   --actor.gradient_checkpointing_enable \
   --ds.packing_samples \
   --vllm.sync_backend nccl \
   --vllm.enforce_eager \
   --vllm.enable_sleep \
   --ds.enable_sleep \
   --train.dynamic_batch_enable \
   --train.max_tokens_per_gpu 16384 \
   --algo.advantage.is_correction_enable

# Enable tensor parallelism for DeepSpeed
#    --ds.tensor_parallel_size 2 \

# Enable Ring-Attention
#    --ds.ring_attn_size 4 \
#    --ds.ring_attn_head_stride 2 \