#!/bin/bash
#SBATCH --job-name=Statistics
#SBATCH --account=
#SBATCH --qos=
#SBATCH --mail-type=NONE
#SBATCH --mail-user=NONE
#SBATCH --cpus-per-gpu=128           
#SBATCH --nodes=1             
#SBATCH --gpus=1
#SBATCH --mem=500g
#SBATCH --time=10:00:00
#SBATCH --output=
#SBATCH --partition=hpg-ai
pwd; hostname; date

module load conda
conda activate graph
 
cd path to this repo

echo "Launching job for script"
python3 4.Statistics.py -s  $1