#!/bin/bash
#SBATCH --job-name=Modeling
#SBATCH --account=
#SBATCH --qos=y
#SBATCH --mail-type=NONE
#SBATCH --mail-user=NONE
#SBATCH --cpus-per-gpu=128           
#SBATCH --nodes=1             
#SBATCH --gpus=1
#SBATCH --mem=1000g
#SBATCH --time=300:00:00
#SBATCH --output=
#SBATCH --partition=
pwd; hostname; date

module load conda
conda activate graph

cd path to this repo

echo "Launching job for script"
python3 Predict_modeling/modeling.py
