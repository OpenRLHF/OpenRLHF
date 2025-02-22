#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=`pwd`


export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH

export TRITON_PTXAS_PATH=/usr/local/cuda-12.4/bin/ptxas                                                                      
export TRITON_CUOBJDUMP_PATH=/usr/local/cuda-12.4/bin/cuobjdump                                                              
export TRITON_NVDISASM_PATH=/usr/local/cuda-12.4/bin/nvdisasm  

deepspeed --module openrlhf.cli.train_ppo \
  --pretrain /app/qi/backup/models/Llama-3-8b-sft-mixture \
  --reward_pretrain /app/qi/backup/models/Llama-3-8b-rm-mixture \
  --save_path ./checkpoint/llama-3-8b-rlhf \
  --save_steps -1 \
  --logging_steps 1 \
  --eval_steps -1 \
  --micro_train_batch_size 1 \
  --train_batch_size 128 \
  --micro_rollout_batch_size 2 \
  --rollout_batch_size 1024 \
  --max_epochs 1 \
  --prompt_max_len 1024 \
  --generate_max_len 1024 \
  --zero_stage 2 \
  --bf16 \
  --actor_learning_rate 5e-7 \
  --critic_learning_rate 9e-6 \
  --init_kl_coef 0.01 \
  --prompt_data OpenRLHF/prompt-collection-v0.1 \
  --input_key context_messages \
  --apply_chat_template \
  --max_samples 100000 \
  --normalize_reward \
  --adam_offload \
  --flash_attn \
  --gradient_checkpointing \
  --use_wandb f3b175fa54df63e7b0592b1bf157744eba49ef44
  

# Support remote reward model (HTTP)
# --remote_rm_url http://localhost:5000/get_reward