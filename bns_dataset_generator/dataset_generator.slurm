#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=dataset_gen
#SBATCH --output=%x.out
#SBATCH --error=%x.err
#SBATCH --time=10:00:00
#SBATCH --partition=submit-gpu
#SBATCH --mem=32GB
#SBATCH --mail-type=END
#SBATCH --mail-user=bhgupta@mit.edu

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# Run dataset generator
/work/submit/bhgupta/miniconda3/envs/mlgw/bin/python3.10 /work/submit/bhgupta/mlgw/dataset_generator.py