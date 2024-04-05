#!/bin/bash
#SBATCH --job-name=cpath
#SBATCH --output=out_cpath_%a.txt
#SBATCH --error=err_cpath_%a.txt
#SBATCH --ntasks=1
#SBATCH --mem=24Gb
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --partition=long
#SBATCH --time=01:00:00
#SBATCH --array=0-8

module load anaconda/3
source /home/mila/p/paria.mehrbod/.bashrc
conda activate pathology

corruptions=("01_focus" "02_jpeg" "04_rotation" "05_flip" "08_bright" "09_contrast" "10_dark_spots" "12_squamous" "13_fat")   

corruption=${corruptions[$SLURM_ARRAY_TASK_ID]}

python 3_Validation_TTA.py --artifact $corruption --cor_path "Corrupted_data/01_case_western_native/"