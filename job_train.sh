#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=train_job.txt
#SBATCH --error=etrain_job.txt
#SBATCH --ntasks=1
#SBATCH --mem=24Gb
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --partition=long
#SBATCH --time=05:00:00


module load anaconda/3
source /home/mila/p/paria.mehrbod/.bashrc
conda activate pathology

CUDA_LAUNCH_BLOCKING=1 python3 train.py