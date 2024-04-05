#!/bin/bash
#SBATCH --job-name=dl
#SBATCH --output=out_dl.txt
#SBATCH --error=err_dl.txt
#SBATCH --mem=8G
#SBATCH --partition=long
#SBATCH --time=02:00:00

echo "https://zenodo.org/records/4789576/files/03_wns_leica_native.tar?download=1
https://zenodo.org/records/4789576/files/04_wns_hama_native.tar?download=1" > links

for URL in $(cat links)
do
  wget -nc $URL
done