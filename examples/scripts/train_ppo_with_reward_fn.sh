# /openrlhf/examples/scripts/reward_func.py
# import torch

# def reward_func(queries, prompts, labels):
#     # queries is prompts + responses
#     # labels is answers
#     print(queries)
#     return torch.randn(len(queries))

set -x 

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/openrlhf"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref.num_nodes 1 \
   --ref.num_gpus_per_node 2 \
   --critic.num_nodes 1 \
   --critic.num_gpus_per_node 2 \
   --actor.num_nodes 1 \
   --actor.num_gpus_per_node 2 \
   --vllm.num_engines 2 \
   --vllm.tensor_parallel_size 2 \
   --train.colocate_actor_ref \
   --actor.model_name_or_path OpenRLHF/Llama-3-8b-sft-mixture \
   --reward.remote_url /openrlhf/examples/python/reward_func.py \
   --ckpt.output_dir /openrlhf/examples/checkpoint/llama3-8b-rlhf \
   --train.micro_batch_size 8 \
   --train.batch_size 128 \
   --rollout.micro_batch_size 16 \
   --rollout.batch_size 1024 \
   --data.max_samples 100000 \
   --train.max_epochs 1 \
   --data.max_len 2048 \
   --ds.zero_stage 3 \
   --ds.param_dtype bf16 \
   --actor.adam.lr 5e-7 \
   --critic.adam.lr 9e-6 \
   --algo.kl.init_coef 0.01 \
   --data.prompt_dataset OpenRLHF/prompt-collection-v0.1 \
   --data.input_key context_messages \
   --data.apply_chat_template \
   --reward.normalize_enable \
   --ds.packing_samples \
   --ds.adam_offload \
   --ds.attn_implementation flash_attention_2 \
   --actor.gradient_checkpointing_enable \
   --logger.wandb.key {wandb_token}

