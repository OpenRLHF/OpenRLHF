set -x

python3 -m openrlhf.cli.train_ppo_ray \
   --ref.num_nodes 1 \
   --ref.num_gpus_per_node 8 \
   --actor.num_nodes 1 \
   --actor.num_gpus_per_node 8 \
   --vllm.num_engines 4 \
   --vllm.tensor_parallel_size 2 \
   --train.colocate_all \
   --vllm.gpu_memory_utilization 0.6 \
   --algo.kl.init_coef 1e-3 \
   --algo.advantage.gamma 1.0 \
   --algo.kl.use_loss \
   --algo.kl.estimator k3 \
   --algo.advantage.estimator group_norm \
   --algo.dynamic_filtering_enable \
   --algo.dynamic_filtering_range 0 1 \
   --actor.eps_clip_low_high 0.2 0.27 \
   --actor.model_name_or_path OpenRLHF/Llama-3-8b-sft-mixture \
   --reward.remote_url /openrlhf/examples/python/reward_func.py \
   --ckpt.output_dir /openrlhf/examples/test_scripts/final/llama3-8b-rlhf \
   --ckpt.path /openrlhf/examples/test_scripts/ckpt/llama3-8b-rlhf \
   --ckpt.save_steps 20 \
   --ckpt.save_hf \
   --train.micro_batch_size 8 \
   --train.batch_size 128 \
   --rollout.micro_batch_size 16 \
   --rollout.batch_size 128 \
   --rollout.n_samples_per_prompt 8 \
   --train.max_epochs 1 \
   --data.max_len 2048 \
   --data.max_samples 20000 \
   --ds.zero_stage 3 \
   --ds.param_dtype bf16 \
   --actor.adam.lr 5e-7 \
   --data.prompt_dataset OpenRLHF/prompt-collection-v0.1 \
   --data.input_key context_messages \
   --data.apply_chat_template \
   --actor.gradient_checkpointing_enable \
   --ds.packing_samples \
   --vllm.sync_backend nccl \
   --vllm.enforce_eager \
   --vllm.enable_sleep \
   --ds.enable_sleep \
   --algo.advantage.is_correction_enable

# You could also try
#   --algo.kl.estimator k2 \
