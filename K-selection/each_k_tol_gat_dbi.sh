#!/bin/bash
#SBATCH --job-name=tol_graphsage_k
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=
#SBATCH --error=".../logs/error_tol_dbi.txt"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=500gb
#SBATCH --time=80:00:00
#SBATCH --output=".../logs/tol_dbi.out"
#SBATCH --partition=hpg-default
#SBATCH --account=guoj1

pwd; hostname; date

module load conda
conda activate graph_model_env
 
echo "Launching job for script"
python3 -u find_gat_tol_k_dbi.py -s $1