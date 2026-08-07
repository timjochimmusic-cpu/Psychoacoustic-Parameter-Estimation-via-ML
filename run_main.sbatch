#!/bin/bash
#SBATCH --job-name=psychoacoustics
#SBATCH --partition=standard
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/beegfs/home/users/t/tjochim/projects/logs/main-%j.out
#SBATCH --error=/beegfs/home/users/t/tjochim/projects/logs/main-%j.err

cd /beegfs/home/users/t/tjochim/projects/Psychoacoustic-Parameter-Estimation-via-ML
source .venv/bin/activate

srun python main.py

