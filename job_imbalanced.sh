#!/bin/bash
#SBATCH --job-name=cpath2
#SBATCH --output=out_cpath2_%a.txt
#SBATCH --error=err_cpath2_%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24Gb
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --partition=long
#SBATCH --time=03:00:00

module load anaconda/3
source /home/mila/p/paria.mehrbod/.bashrc
conda activate pathology


python 4_Validation_TTA_imbalanced.py --model_name "TvN_350_SN_D256_v2_Ep1_fullmodel.pth" --batch_size 64 --cor_path "02_training_native" --exp_type "imbalanced_experiments"
