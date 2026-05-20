set -x

export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo

export HSA_NO_SCRATCH_RECLAIM=1
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

ray stop --force 2>/dev/null || true
sleep 5

TIMESTAMP=$(date "+%Y.%m.%d-%H.%M.%S")
LOG=./checkpoint/llama3-8b-rlhf/llama3-8b-rlhf-${TIMESTAMP}.log

mkdir -p ./checkpoint/llama3-8b-rlhf


python3 -m openrlhf.cli.train_ppo_ray \
   --ref.num_nodes 1 \
   --ref.num_gpus_per_node 8 \
   --reward.num_nodes 1 \
   --reward.num_gpus_per_node 8 \
   --critic.num_nodes 1 \
   --critic.num_gpus_per_node 8 \
   --actor.num_nodes 1 \
   --actor.num_gpus_per_node 8 \
   --vllm.num_engines 1 \
   --vllm.tensor_parallel_size 8 \
   --train.colocate_all \
   --vllm.gpu_memory_utilization 0.6 \
   --actor.model_name_or_path ./shared/Llama-3-8b-sft-mixture \
   --reward.model_name_or_path ./shared/Llama-3-8b-rm-700k \
   --ckpt.output_dir ./output/llama3-8b-rlhf \
   --ckpt.path ./output/llama3-8b-rlhf \
   --train.batch_size 128 \
   --rollout.batch_size 1024 \
   --rollout.n_samples_per_prompt 1 \
   --train.max_epochs 1 \
   --data.max_len 2048 \
   --data.max_samples 80000 \
   --ds.zero_stage 3 \
   --ds.param_dtype bf16 \
   --actor.adam.lr 5e-7 \
   --critic.adam.lr 9e-6 \
   --algo.kl.init_coef 0.01 \
   --data.prompt_dataset ./shared/prompt-collection-v0.1 \
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
   --algo.advantage.is_correction_enable 2>&1 | tee $LOG
