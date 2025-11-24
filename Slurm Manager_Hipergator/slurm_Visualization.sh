#!/bin/bash
#SBATCH --job-name=Visualization
#SBATCH --account=
#SBATCH --qos=
#SBATCH --mail-type=NONE
#SBATCH --mail-user=NONE
#SBATCH --cpus-per-gpu=128           
#SBATCH --nodes=1             
#SBATCH --gpus=1
#SBATCH --mem=1024g
#SBATCH --time=20:00:00
#SBATCH --output=
#SBATCH --partition=hpg-ai
pwd; hostname; date

module load conda
conda activate graph

cd path to this repo
 
echo "Launching job for script"
python3 3.Visualization.py -s  $1
