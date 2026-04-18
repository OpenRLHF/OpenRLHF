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
   --vllm_num_engines 2 \
   --vllm_tensor_parallel_size 2 \
   --colocate_actor_ref \
   --actor.model_name_or_path OpenRLHF/Llama-3-8b-sft-mixture \
   --reward.remote_url /openrlhf/examples/python/reward_func.py \
   --save_path /openrlhf/examples/checkpoint/llama3-8b-rlhf \
   --micro_train_batch_size 8 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 16 \
   --rollout_batch_size 1024 \
   --max_samples 100000 \
   --max_epochs 1 \
   --max_len 2048 \
   --zero_stage 3 \
   --param_dtype bf16 \
   --actor.adam.lr 5e-7 \
   --critic.adam.lr 9e-6 \
   --init_kl_coef 0.01 \
   --prompt_data OpenRLHF/prompt-collection-v0.1 \
   --input_key context_messages \
   --apply_chat_template \
   --reward.normalize \
   --packing_samples \
   --adam_offload \
   --actor.attn_implementation flash_attention_2 \
   --actor.gradient_checkpointing \
   --use_wandb {wandb_token}

