#!/bin/bash

#SBATCH -p { partition }              
#SBATCH -A { account }
#SBATCH -J { jobname }
#SBATCH -N 1                      # 64x8x4
#SBATCH -t 0-00:30:00             # wall time
#SBATCH --ntasks-per-node=1       # tasks per node
#SBATCH --exclusive                # exclusive node access
#SBATCH --mem=0                    # all mem avail
#SBATCH --mail-type=FAIL           # only send email on failure
#SBATCH --overcommit               # needed for pytorch
#SBATCH --output=out.log

# should be modified to train_sft_llama.sh, train_rm_llama.sh, train_dpo_llama, etc.
readonly training_script="train_dpo_llama.sh" 
readonly GPUS_PER_NODE=8

readonly PROJECT_PATH=$(cd ../../; pwd)
readonly IMAGE_NAME="nvcr.io/nvidia/pytorch:25.02-py3"
readonly JOBLOG="$(pwd)/logs/$training_script-$SLURM_JOB_ID.log"
mkdir logs

# Job start
echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} started ..." &>> ${JOBLOG}

# load training commands
source ./${training_script} slurm
echo training_commands &>> ${JOBLOG}
echo $training_commands &>> ${JOBLOG}

# master addr and port
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9901

srun --container-image="$IMAGE_NAME" \
    --container-mounts="$PROJECT_PATH:/openrlhf,$HOME/.cache:/root/.cache" \
    bash -c " cd /openrlhf; pip install . ; torchrun \
--nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
--master_addr $MASTER_ADDR --master_port $MASTER_PORT -m ${training_commands}" &>> ${JOBLOG}

echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} stopped ..." &>> ${JOBLOG}