set -x 
export PATH=$HOME/.local/bin/:$PATH

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json='{"working_dir": "/openllama2", "pip": "/openllama2/requirements.txt"}' \
    -- python3 examples/train_ppo_ray.py \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 1 \
    --reward_num_nodes 1 \
    --reward_num_gpus_per_node 1 \
    --critic_num_nodes 1 \
    --critic_num_gpus_per_node 2 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 4 \
    --pretrain meta-llama/Llama-2-7b-hf \
    --critic_pretrain meta-llama/Llama-2-7b-hf \
    --reward_model_path /openllama2/examples/test_scripts/ckpt/7b_llama/rm_model_anthropic_oasst_lmsys_webgpt.pt \
    --sft_model_path /openllama2/examples/test_scripts/ckpt/7b_llama/sft_model_ocra.pt \
    --save_path /openllama2/examples/test_scripts/ckpt/7b_llama \
    --micro_train_batch_size 4 \
    --train_batch_size 128 \
    --micro_rollout_batch_size 8 \
    --rollout_batch_size 1024 \
    --max_epochs 1 \
    --prompt_max_len 1024 \
    --generate_max_len 1024 \
    --zero_stage 2 \
    --bf16 \
    --actor_learning_rate 5e-7 \
    --critic_learning_rate 9e-6 \
    --inference_tp_size 1 \
    --init_kl_coef 0.01 \
    --prompt_data Open-Orca/OpenOrca,Dahoas/full-hh-rlhf,tasksource/oasst1_pairwise_rlhf_reward \
    --prompt_data_probs 0.4,0.5,0.1 \
    --max_samples 80000 \
    --normalize_reward \
    --actor_init_on_gpu \
    --gradient_checkpointing