#!/bin/bash -l

#SBATCH --job-name=benchmark_cnn
#SBATCH --time=00:05:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
##SBATCH --partition=debug
#SBATCH --constraint=gpu
#SBATCH --account=usup

module load daint-gpu
module load PyTorch

srun python cnn_distr.py
