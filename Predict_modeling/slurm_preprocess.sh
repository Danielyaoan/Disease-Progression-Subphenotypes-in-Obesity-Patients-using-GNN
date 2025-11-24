#!/bin/bash
#SBATCH --job-name=Preprocess
#SBATCH --account=y
#SBATCH --qos=
#SBATCH --mail-type=NONE
#SBATCH --mail-user=NONE
#SBATCH --cpus-per-gpu=128           
#SBATCH --nodes=1             
#SBATCH --gpus=1
#SBATCH --mem=50g
#SBATCH --time=10:00:00
#SBATCH --output=
#SBATCH --partition=
pwd; hostname; date

module load conda
conda activate graph
 
echo "Launching job for script"
python3 preprocess.py
