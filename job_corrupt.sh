#!/bin/bash
#SBATCH --job-name=cpath
#SBATCH --output=out_cpath_%a.txt
#SBATCH --error=err_cpath_%a.txt
#SBATCH --ntasks=1
#SBATCH --mem=24Gb
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --partition=long
#SBATCH --time=02:00:00

module load anaconda/3
source /home/mila/p/paria.mehrbod/.bashrc
conda activate pathology

# paths=("02_training_native")

# "07_elastic", "12_squamous","13_fat","15_stain_scheme", "10_dark_spots", "11_synthetic_thread","02_jpeg","03_rotate", "05_flip","08_bright","09_contrast" $SLURM_ARRAY_TASK_ID
# path=${paths[$SLURM_ARRAY_TASK_ID]}

CUDA_LAUNCH_BLOCKING=1 python Corrupt_data.py --source_path "02_training_native"