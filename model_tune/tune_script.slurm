#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=2
#SBATCH --job-name=tune_script
#SBATCH --output=%x.out
#SBATCH --error=%x.err
#SBATCH --time=6:00:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --mem-per-cpu=16GB
#SBATCH --mail-type=END
#SBATCH --mail-user=bhgupta@mit.edu

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

#module load anaconda3/2022.05-x86_64 

#head_node=$(hostname | cut -d. -f1)
#head_node_ip=$(hostname --ip-address)
#port=6379

# echo "STARTING HEAD at $head_node"
# echo "Head node IP: $head_node_ip"
# srun --nodes=1 --ntasks=1 -w "$head_node" \
#     /orcd/home/002/bhgupta/.conda/envs/mlgw/bin/ray start --head --node-ip-address="$head_node_ip" --port=$port \
#     --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus 1 --block &
# sleep 10

# echo "HEAD NODE STARTED"

# worker_num=$(($SLURM_JOB_NUM_NODES - 1))
# srun --ntasks=$worker_num --nodes=$worker_num --ntasks-per-node=1 --exclude=$head_node \
#     /orcd/home/002/bhgupta/.conda/envs/mlgw/bin/ray start --address $head_node_ip:$port --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus 1 --block &

# echo "SLEEPING FOR 60s BEFORE CALLING SCRIPT"
# sleep 60


# echo "ASSUMING HEAD AND WORKERS HAVE STARTED...CALLING SCRIPT"
/orcd/home/002/bhgupta/.conda/envs/mlgw/bin/python3.10 /orcd/scratch/orcd/007/bhgupta/mlgw/tune_script.py
