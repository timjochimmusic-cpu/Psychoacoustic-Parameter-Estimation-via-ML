#!/bin/bash
#SBATCH --job-name=plot_stats
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/beegfs/home/users/t/tjochim/projects/Psychoacoustic-Parameter-Estimation-via-ML/logs/main-%j.out
#SBATCH --error=/beegfs/home/users/t/tjochim/projects/Psychoacoustic-Parameter-Estimation-via-ML/logs/main-%j.err

cd /beegfs/home/users/t/tjochim/projects/Psychoacoustic-Parameter-Estimation-via-ML

source .venv/bin/activate

python training_data_visualization/plot_dataset_statistics.py