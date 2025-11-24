#!/bin/bash
#SBATCH --job-name=model_training
#SBATCH --account=
#SBATCH --qos=
#SBATCH --mail-type=NONE
#SBATCH --mail-user=NONE
#SBATCH --cpus-per-gpu=32           
#SBATCH --nodes=1             
#SBATCH --gpus=1
#SBATCH --mem=300g
#SBATCH --time=100:00:00
#SBATCH --output=
#SBATCH --partition=hpg-ai
pwd; hostname; date

module load conda
conda activate graph

cd path to this repo

echo "Launching job for script"
python3 1.Model_tuning.py -s  $1
