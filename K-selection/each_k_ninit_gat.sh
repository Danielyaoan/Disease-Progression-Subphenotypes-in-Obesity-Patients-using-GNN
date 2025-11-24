#!/bin/bash
#SBATCH --job-name=tol_graphsage_k
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=
#SBATCH --error=".../logs/error_ninit_sil.txt"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=300gb
#SBATCH --time=80:00:00
#SBATCH --output=".../logs/ninit_sil.out"
#SBATCH --partition=
#SBATCH --account=

pwd; hostname; date

module load conda
conda activate graph_model_env
 
echo "Launching job for script"
python3 -u find_gat_nint_k.py -s $1