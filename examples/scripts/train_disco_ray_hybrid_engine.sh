set -x

python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 8 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 8 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 8 \
   --vllm_num_engines 4 \
   --vllm_tensor_parallel_size 2 \
   --colocate_all_models \
   --vllm_gpu_memory_utilization 0.6 \
   --init_kl_coef 0.01 \
   --gamma 1.0 \
   --advantage_estimator disco \
   --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
   --reward_pretrain OpenRLHF/Llama-3-8b-rm-700k \
   --save_path /openrlhf/examples/test_scripts/final/llama3-8b-disco \
   --ckpt_path /openrlhf/examples/test_scripts/ckpt/llama3-8b-disco \
   --save_hf_ckpt \
   --micro_train_batch_size 8 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 16 \
   --rollout_batch_size 128 \
   --n_samples_per_prompt 1 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --max_samples 100000 \
   --generate_max_len 1024 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --prompt_data OpenRLHF/prompt-collection-v0.1 \
   --input_key context_messages \
   --apply_chat_template \
   --normalize_reward \
   --gradient_checkpointing \
   --packing_samples \
   --vllm_sync_backend nccl \
   --enforce_eager \
   --vllm_enable_sleep \
   --deepspeed_enable_sleep \
   --disco_type "disco" \
   --disco_beta 1.0 \
   --disco_delta 0.01 \
   --disco_tau 0.1 \
   --disco_scoring_func "log_l" \
   # --disco_reward_threshold 0.5 # Optional: uncomment to set a fixed threshold, otherwise median is used
   # The following GRPO specific args are removed or not applicable to DisCO by default:
   # --use_kl_loss  # DisCO has its own KL mechanism integrated, not the PPO-style KL with reference model.
   # --kl_estimator k3 # Related to use_kl_loss
   # --n_samples_per_prompt 8 # DisCO typically uses 1 sample per prompt for its objective, but can be >1 if needed.
                               # For this script, set to 1. If using >1, ensure reward processing logic is appropriate.

# Note: For DisCO, the 'critic' model and its associated parameters (like critic_learning_rate)
# might not be directly used in the DisCO objective if following the original paper closely.
# The PPO trainer structure might still initialize it. Depending on the final DisCO implementation details
# in the actor, the critic's role might be vestigial or repurposed.
# The `init_kl_coef` is used for the KL penalty in DisCO (beta), not for PPO's KL divergence with a reference model.
# For DisCO, the KL is between the current policy and the behavior policy (old policy from replay buffer).
# The `disco_beta` argument in the DisCO loss function corresponds to the KL penalty coefficient.
# The `init_kl_coef` here might be confusing; `disco_beta` is the primary controller for KL penalty in DisCO loss.
# For clarity, `init_kl_coef` is kept as it's part of the base PPO args, but its role in DisCO is defined by `disco_beta`.
# The PPO KL controller (`AdaptiveKLController` or `FixedKLController`) is not directly used by the DisCO loss logic.
# DisCO's KL penalty is calculated directly within its loss function.
# `gamma` is generally 1.0 for sequence-level objectives like DisCO.
# `normalize_reward` might not be directly applicable to binary rewards in DisCO.
# `use_kl_loss` from PPO (KL against a fixed reference model) is not used; DisCO's KL is against the behavior policy.
# `kl_estimator` is also not used for the same reason.
# `n_samples_per_prompt` is set to 1 as DisCO's formulation is often per (positive_sample, negative_sample_set)
# or based on expectations over single samples. If using multiple samples, ensure the DisCO loss and data handling are compatible.
# The current DisCO loss implementation expects per-sample binary rewards.
# The `advantage_estimator` is set to "disco" which triggers the DisCO loss logic.
# The `disco_type` further specifies the DisCO variant.
# Ensure `reward_pretrain` points to a model that gives good continuous scores for thresholding/median split.
# `OpenRLHF/Llama-3-8b-rm-700k` is a reasonable starting point.
# The `save_path` and `ckpt_path` are updated to include "disco".
# `actor_learning_rate` and other general hyperparameters are kept similar to GRPO as a starting point.
# `critic_learning_rate` might be irrelevant if the critic is not trained with DisCO.
# `packing_samples` and `gradient_checkpointing` are kept for efficiency.
# `vllm` related flags are kept for hybrid engine setup.
# `zero_stage 3` and `bf16` are kept for training efficiency.
# `prompt_data`, `input_key`, `apply_chat_template` are dataset specific.
# `generate_max_len`, `prompt_max_len` are sequence length controls.
# `max_samples`, `max_epochs` control training duration.
# `micro_train_batch_size`, `train_batch_size`, `micro_rollout_batch_size`, `rollout_batch_size` control batching.
# `colocate_all_models`, `vllm_gpu_memory_utilization`, `vllm_sync_backend`, `enforce_eager`, `vllm_enable_sleep`, `deepspeed_enable_sleep` are system/performance settings.
