#!/bin/bash

#SBATCH --job-name=MTE
#SBATCH -n 1
#SBATCH --partition=bii
#SBATCH --account=bii_nssac
##SBATCH --reservation=maintenance
##SBATCH --qos=bii-unlimited
#SBATCH --output=log
#SBATCH --time=20:00:00
#SBATCH --mem 50G
export CONDA_ROOT="/home/$USER/miniconda3"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate idtxl
python slurm_multivariate_TE_49_regions_cases.py $1
