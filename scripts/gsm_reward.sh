#!/bin/bash
#SBATCH --job-name=gsm_rewards
#SBATCH --account=kempner_dev
#SBATCH --partition=kempner_h100
#SBATCH --output /n/holylfs06/LABS/kempner_dev/Lab/nikhilanand/agents/AgentsOpenRLHF/serverlogs/%A.log
#SBATCH --error /n/holylfs06/LABS/kempner_dev/Lab/nikhilanand/agents/AgentsOpenRLHF/serverlogs/error_%j.out
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:2                             
#SBATCH --ntasks-per-node=1                                  
#SBATCH --cpus-per-task=32               
#SBATCH --mem=200GB

echo "Running on host: $(hostname)"

source ~/.bashrc
conda deactivate
conda activate openrlhf_202507

module load cuda/12.4.1-fasrc01
module load gcc/14.2.0-fasrc01
module load cmake/3.31.6-fasrc01
module load cudnn

export LD_LIBRARY_PATH=/n/sw/helmod-rocky8/apps/Core/cuda/12.4.1-fasrc01/cuda/lib64:$LD_LIBRARY_PATH

python3 -m openrlhf.cli.serve_gsm_rm