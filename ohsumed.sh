#!/bin/sh
#SBATCH --job-name=ohsumed # Job name
#SBATCH --ntasks=4 # Run on a single CPU
#SBATCH --time=23:50:00 # Time limit hrs:min:sec
#SBATCH --output=test_job%j%x.out # Standard output and error log
#SBATCH --gres=gpu:1
#SBATCH --partition=cl1_48h-1G
python3 main.py --lr .0001 --num_layers 3 --num_mlp_layers 1 --hidden_dim 400 --final_dropout .5 --configfile ohsumed --num_heads 1
