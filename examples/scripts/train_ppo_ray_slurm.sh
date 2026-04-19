#!/bin/bash

#SBATCH -p { partition }
#SBATCH -A { account }
#SBATCH -J { jobname }
#SBATCH -N 2                       # 64x8x4
#SBATCH -t {LIMIT_TIME}            # wall time
#SBATCH --ntasks-per-node=1        # tasks per node
#SBATCH --exclusive                # exclusive node access
#SBATCH --mem=0                    # all mem avail
#SBATCH --mail-type=FAIL           # only send email on failure
#SBATCH --overcommit               # needed for pytorch

# project settings
OPENRLHF_PATH=<OPENRLHF_ROOT_PATH>
MOUNT="$OPENRLHF_PATH:/openrlhf,$HOME/.cache:/root/.cache"
IMAGE_NAME="nvcr.io/nvidia/pytorch:25.11-py3"
RAY_VERSION=2.12.0

JOBLOG="$(realpath .)/train_ppo_llama_ray-$SLURM_JOB_ID.log"
echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} started ..." &>> ${JOBLOG}

# launch ray daemon
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST") # Getting the node names
nodes_array=( $nodes )
node_1=${nodes_array[0]}
ip=$node_1

port=6379
ip_head=$ip:$port
export ip_head
echo "IP Head: $ip_head"  &>> ${JOBLOG}

echo "STARTING HEAD at $node_1"  &>> ${JOBLOG}
srun --nodes=1 --ntasks=1 -w "$node_1" --container-image="$IMAGE_NAME" --container-mounts="$MOUNT" bash -c \
    "pip install ray[default]==$RAY_VERSION \
    && /root/.local/bin/ray start --head --node-ip-address=$ip --port=$port --block" &>> ${JOBLOG} &
sleep 10s

worker_num=$((SLURM_JOB_NUM_NODES)) #number of nodes other than the head node
for ((i = 1; i < worker_num; i++)); do
node_i=${nodes_array[$i]}
echo "STARTING WORKER $i at $node_i"  &>> ${JOBLOG}
srun --nodes=1 --ntasks=1 -w "$node_i" --container-image="$IMAGE_NAME" --container-mounts="$MOUNT" bash -c \
    "pip install ray[default]==$RAY_VERSION \
    && /root/.local/bin/ray start --address $ip_head --block" &>> ${JOBLOG} &
sleep 1s;
done

sleep 30s

# ===== submit ray job =====
# Job start
srun --overlap --nodes=1 --ntasks=1 -w "$node_1" --container-image="$IMAGE_NAME" --container-mounts="$MOUNT" bash -c \
"pip install ray[default]==$RAY_VERSION \
&& /root/.local/bin/ray job submit --address=http://localhost:8265 \
    --runtime-env-json='{\"working_dir\": \"/openrlhf\", \"pip\": \"/openrlhf/requirements.txt\"}' \
    -- python3 -m openrlhf.cli.train_ppo_ray \
    --ref.num_nodes 1 \
    --ref.num_gpus_per_node 4 \
    --reward.num_nodes 1 \
    --reward.num_gpus_per_node 4 \
    --critic.num_nodes 1 \
    --critic.num_gpus_per_node 4 \
    --actor.num_nodes 1 \
    --actor.num_gpus_per_node 4 \
    --vllm.num_engines 4 \
    --vllm.tensor_parallel_size 2 \
    --train.colocate_critic_reward \
    --train.colocate_actor_ref \
    --actor.model_name_or_path OpenRLHF/Llama-3-8b-sft-mixture \
    --reward.model_name_or_path OpenRLHF/Llama-3-8b-rm-mixture \
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
    --ds.adam_offload \
    --ds.packing_samples \
    --vllm.sync_backend nccl \
    --actor.gradient_checkpointing_enable \
    --logger.wandb.key {wandb_token}" &>> ${JOBLOG}

echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} stopped ..." &>> ${JOBLOG}