#!/bin/sh
#SBATCH --job-name=R52 # Job name
#SBATCH --ntasks=4 # Run on a single CPU
#SBATCH --time=23:50:00 # Time limit hrs:min:sec
#SBATCH --output=test_job%j%x.out # Standard output and error log
#SBATCH --gres=gpu:1
#SBATCH --partition=cl1_48h-1G
python3 main.py --lr .001 --num_layers 3 --num_mlp_layers 1 --hidden_dim 100 --learn_eps --configfile R52
