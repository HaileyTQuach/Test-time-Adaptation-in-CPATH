#!/bin/bash
#SBATCH --job-name=cpath
#SBATCH --output=out_cpath_%a.txt
#SBATCH --error=err_cpath_%a.txt
#SBATCH --nodes=1
#SBATCH --gpus-per-node=rtx8000:1
#SBATCH --mem=48Gb
#SBATCH --cpus-per-task=1
#SBATCH --partition=long
#SBATCH --time=00:30:00
#SBATCH --array=0-9

module load anaconda/3
source /home/mila/p/paria.mehrbod/.bashrc
conda activate pathology

corruptions=("00_original" "01_focus" "02_jpeg" "04_rotation" "05_flip" "08_bright" "09_contrast" "10_dark_spots" "12_squamous" "13_fat")   

corruption=${corruptions[$SLURM_ARRAY_TASK_ID]}

python 3_Validation_TTA.py --artifact $corruption --cor_path "02_training_native/" --model_name "TvN_350_SN_D256_Initial_Ep7_fullmodel.pth" --exp_type "corr_experiments"