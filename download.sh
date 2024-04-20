#!/bin/bash
#SBATCH --job-name=dl
#SBATCH --output=out_dl.txt
#SBATCH --error=err_dl.txt
#SBATCH --mem=8G
#SBATCH --partition=long
#SBATCH --time=03:00:00

echo "https://zenodo.org/records/4904569/files/05_wns_glis_native.tar?download=1
https://zenodo.org/records/4904569/files/06_ukk_native.tar?download=1
" > links

for URL in $(cat links)
do
  wget -nc $URL
done