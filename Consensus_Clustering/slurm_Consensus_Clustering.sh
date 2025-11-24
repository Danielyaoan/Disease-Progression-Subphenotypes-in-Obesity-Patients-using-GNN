#!/bin/bash
#SBATCH --job-name=Consensus_Clustering
#SBATCH --account=
#SBATCH --qos=
#SBATCH --mail-type=NONE
#SBATCH --mail-user=NONE
#SBATCH --cpus-per-gpu=128           
#SBATCH --nodes=1             
#SBATCH --gpus=1
#SBATCH --mem=1000g
#SBATCH --time=60:00:00
#SBATCH --output=
#SBATCH --partition=hpg-ai
pwd; hostname; date

module load conda
conda activate graph
 
echo "Launching job for script"
python3 Consensus_Clustering_TS_kmeans.py
