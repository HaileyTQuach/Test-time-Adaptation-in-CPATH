#!/bin/bash
#SBATCH --job-name=cpath_batch_size
#SBATCH --output=out_cpath_batch_size%a.txt
#SBATCH --error=err_cpath_batch_size%a.txt
#SBATCH --nodes=1
#SBATCH --gpus-per-node=rtx8000:1
#SBATCH --mem=48Gb
#SBATCH --cpus-per-task=1
#SBATCH --partition=long
#SBATCH --time=01:00:00
#SBATCH --array=0-5

module load anaconda/3
source /home/mila/p/paria.mehrbod/.bashrc
conda activate pathology

corruptions=("00_original" "01_focus" "09_contrast" "10_dark_spots" "12_squamous" "13_fat")   # "02_jpeg" "04_rotation" "05_flip" "08_bright"

# batch_sizes=(64)

# Calculate the index for the current batch size and corruption
# batch_size_index=$((SLURM_ARRAY_TASK_ID / ${#corruptions[@]}))
# corruption_index=$((SLURM_ARRAY_TASK_ID % ${#corruptions[@]}))

# corruption=${corruptions[$corruption_index]}
# batch_size=${batch_sizes[$batch_size_index]}

corruption=${corruptions[$SLURM_ARRAY_TASK_ID]}

# echo $batch_size
echo $corruption
python 3_Validation_TTA.py --artifact $corruption --cor_path "02_training_native/" --model_name "TvN_350_SN_D256_Initial_Ep7_fullmodel.pth" --exp_type "batchsize_experiments" --batch_size 64
