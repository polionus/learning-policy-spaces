#!/bin/bash
#SBATCH --account=aip-lelis      # Replace with your PI's account
#SBATCH --cpus-per-task=2              # Or remove if CPU-only
#SBATCH --mem=4G
#SBATCH --time=23:59:59
#SBATCH --job-name=my_exp

module load python/3.10  # or whatever you use
source .venv/bin/activate  # If using virtualenv/conda

# =====================
# RUN YOUR SCRIPT HERE
# =====================

python main_join_dataset.py --num_progs 100 --seed 0 --save 
