#!/bin/bash
#SBATCH --job-name=Clustering
#SBATCH --account=
#SBATCH --qos=
#SBATCH --mail-type=NONE
#SBATCH --mail-user=NONE
#SBATCH --cpus-per-gpu=128           
#SBATCH --nodes=1             
#SBATCH --gpus=1
#SBATCH --mem=500g
#SBATCH --time=30:00:00
#SBATCH --output=
#SBATCH --partition=hpg-ai
pwd; hostname; date

module load conda
conda activate graph
 
cd path to this repo

echo "Launching job for script"
python3 2.Clustering_TS_kmeans.py -s  $1