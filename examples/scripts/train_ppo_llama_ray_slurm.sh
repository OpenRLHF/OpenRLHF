#!/bin/bash

#SBATCH -p {PARTITION}
#SBATCH -A {ACCOUNT}
#SBATCH -J {ACCOUNT}-openrlhf
#SBATCH -N 2                       # 64x8x4
#SBATCH -t {LIMIT_TIME}            # wall time
#SBATCH --ntasks-per-node=1        # tasks per node
#SBATCH --exclusive                # exclusive node access
#SBATCH --mem=0                    # all mem avail
#SBATCH --mail-type=FAIL           # only send email on failure
#SBATCH --overcommit               # needed for pytorch
#SBATCH --output=out.log


# project settings
PROJECT_PATH=$(cd ../../; realpath .)
IMAGE_NAME="nvcr.io/nvidia/pytorch:24.02-py3"
MOUNT="$PROJECT_PATH:/openrlhf,$HOME/.cache:/root/.cache,/dev/null:/root/.bashrc"

JOBLOG="$(realpath .)/logs/train_ppo_llama_ray-$SLURM_JOB_ID.log"
mkdir logs
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
  && pip install ray[default]==2.12.0 \
  && /root/.local/bin/ray start --head --node-ip-address=$ip --port=$port --block" &>> ${JOBLOG} &
sleep 10s

worker_num=$((SLURM_JOB_NUM_NODES)) #number of nodes other than the head node
for ((i = 1; i < worker_num; i++)); do
  node_i=${nodes_array[$i]}
  echo "STARTING WORKER $i at $node_i"  &>> ${JOBLOG}
  srun --nodes=1 --ntasks=1 -w "$node_i" --container-image="$IMAGE_NAME" --container-mounts="$MOUNT" bash -c \
    && pip install ray[default]==2.12.0 \
    && /root/.local/bin/ray start --address "$ip_head" --block" &>> ${JOBLOG} &
  sleep 1s;
done

sleep 30s

# ===== submit ray job =====
# Job start
srun --overlap --nodes=1 --ntasks=1 -w "$node_1" --container-image="$IMAGE_NAME" --container-mounts="$MOUNT" bash -c \
  "pip install ray[default]==2.10.0 \
  && /root/.local/bin/ray job submit --address=http://localhost:8265 \
    --runtime-env-json='{\"working_dir\": \"/openrlhf\", \"pip\": \"/openrlhf/requirements.txt\"}' \
    -- python3 examples/train_ppo_ray.py \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 2 \
    --reward_num_nodes 1 \
    --reward_num_gpus_per_node 2 \
    --critic_num_nodes 1 \
    --critic_num_gpus_per_node 4 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 8 \
    --pretrain OpenLLMAI/Llama-2-13b-sft-model-ocra-500k \
    --reward_pretrain OpenLLMAI/Llama-2-13b-rm-anthropic_hh-lmsys-oasst-webgpt \
    --save_path /openrlhf/examples/test_scripts/ckpt/13b_llama \
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
    --init_kl_coef 0.01 \
    --prompt_data Open-Orca/OpenOrca,Dahoas/full-hh-rlhf,tasksource/oasst1_pairwise_rlhf_reward \
    --prompt_data_probs 0.4,0.5,0.1 \
    --max_samples 80000 \
    --normalize_reward \
    --adam_offload \
    --flash_attn \
    --gradient_checkpointing \
    --use_wandb {wandb_token}" &>> ${JOBLOG}


echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} stopped ..." &>> ${JOBLOG}