#!/bin/bash -l

#SBATCH --job-name=benchmark_cnn
#SBATCH --time=00:05:00
#SBATCH --nodes=512
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
##SBATCH --partition=debug
#SBATCH --constraint=gpu
#SBATCH --account=usup

. /scratch/snx3000/class101/course/course-environ.sh

srun python cnn_distr.py
