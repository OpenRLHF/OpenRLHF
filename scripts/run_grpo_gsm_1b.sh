#!/bin/bash
#SBATCH --job-name="test_grpo"
#SBATCH --account=kempner_dev
#SBATCH --output /n/holylfs06/LABS/kempner_dev/Lab/nikhilanand/agents/AgentsOpenRLHF/run_logs/%A.log
#SBATCH --error /n/holylfs06/LABS/kempner_dev/Lab/nikhilanand/agents/AgentsOpenRLHF/run_logs/error_%j.out
#SBATCH --export=ALL
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=96
#SBATCH --mem=800GB
#SBATCH --exclusive
#SBATCH --time=0-10:00:00
#SBATCH --partition=kempner_eng


source ~/.bashrc
conda deactivate
#conda activate openrlhf_202507
conda activate openrlhf_test


module purge


module load cuda/12.4.1-fasrc01
module load gcc/12.2.0-fasrc01
module load cmake/3.31.6-fasrc01
module load cudnn

export WANDB_MODE=online
export LD_LIBRARY_PATH=/n/sw/helmod-rocky8/apps/Core/cuda/12.4.1-fasrc01/cuda/lib64:$LD_LIBRARY_PATH

export PYTHONPATH=/n/holylfs06/LABS/kempner_dev/Lab/nikhilanand/agents/AgentsOpenRLHF:$PYTHONPATH

rm -rf ~/.cache/vllm
export GLOO_SOCKET_IFNAME=$(ip -4 addr | awk '/inet/ && !/127.0.0.1/ {print $NF; exit}')
echo "Using network interface for Gloo: $GLOO_SOCKET_IFNAME"

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1


if [[ -z $SLURM_GPUS_ON_NODE ]]; then
    RAY_NUM_GPUS=0
else
    RAY_NUM_GPUS=$SLURM_GPUS_ON_NODE
fi

head_port=`comm -23 <(seq 15000 20000 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1`
nodes=`scontrol show hostnames $SLURM_JOB_NODELIST`
nodes_array=( $nodes )
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo "Head node: $head_node"
echo "Head node ip: $head_node_ip"
echo "Head port: $head_port"
export RAY_HEAD_ADDR="$head_node_ip:$head_port"
echo "Head address: $RAY_HEAD_ADDR"

echo "Starting Ray head on $head_node"
srun -N 1 -n 1 -w "$head_node" ray start --head --node-ip-address="$head_node_ip" --temp-dir /tmp/$USER/$SLURM_JOB_ID/ray \
    --port=$head_port --num-cpus $SLURM_CPUS_PER_TASK --num-gpus $RAY_NUM_GPUS --min-worker-port 20001 --max-worker-port 30000 --block &

# wait for head node to start
sleep 20

export GLOO_SOCKET_IFNAME=$(ip -4 addr | awk '/inet/ && !/127.0.0.1/ {print $NF; exit}')
echo "Using network interface for Gloo: $GLOO_SOCKET_IFNAME"

export MASTER_ADDR=$head_node_ip
export HOSTNAME=$(hostname)


# start ray on the rest of the nodes
worker_num=$((SLURM_NNODES - 1))
for (( i = 1; i <= worker_num; i++ )); do
    node=${nodes_array[$i]}
    echo "Starting Ray worker on $node"
    srun -N 1 -n 1 -w "$node" ray start --address="$RAY_HEAD_ADDR" \
        --num-cpus $SLURM_CPUS_PER_TASK --num-gpus $RAY_NUM_GPUS --min-worker-port 20001 --max-worker-port 30000 --block &
    sleep 10
done

export RAY_ADDRESS="$RAY_HEAD_ADDR"


srun --overlap -N 1 -n 1 -w "$head_node" ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json='{"working_dir": "/n/netscratch/kempner_dev/Everyone/nikhilanand/openrlhf_logs_and_artifacts_202507_working_dir"}' \
  -- python3 -m openrlhf.cli.train_ppo_ray \
  --ref_num_nodes 1 \
  --ref_num_gpus_per_node 4 \
  --colocate_actor_ref \
  --actor_num_nodes 1 \
  --actor_num_gpus_per_node 4 \
  --pretrain /n/holylabs/LABS/kempner_dev/Users/nikhilanand/Llama-3-8B-Instruct-HF \
  --remote_rm_url http://holygpu8a10301.rc.fas.harvard.edu:5000/get_reward \
  --save_path /n/netscratch/kempner_dev/Everyone/nikhilanand/openrlhf_logs_and_artifacts_202507 \
  --vllm_num_engines 2 \
  --vllm_tensor_parallel_size 2 \
  --micro_train_batch_size 8 \
  --train_batch_size 128 \
  --micro_rollout_batch_size 16 \
  --rollout_batch_size 1024 \
  --max_samples 100000 \
  --logging_steps 1 \
  --eval_steps 100 \
  --max_epochs 1 \
  --num_episodes 1 \
  --prompt_max_len 2048 \
  --generate_max_len 1024 \
  --n_samples_per_prompt 8 \
  --zero_stage 3 \
  --bf16 \
  --actor_learning_rate 5e-7 \
  --advantage_estimator group_norm \
  --eps_clip 0.0 \
  --prompt_data 'openai/gsm8k@main' \
  --input_key question \
  --label_key answer \
  --normalize_reward \
  --packing_samples \
  --adam_offload \
  --flash_attn \
  --gradient_checkpointing \
  --use_wandb True \
  --wandb_project openrlhf_test_202507 \
  --wandb_run_name multinode_grpo \
  --apply_chat_template \
  