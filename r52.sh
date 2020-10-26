#!/bin/sh
#SBATCH --job-name=GNN # Job name
#SBATCH --ntasks=4 # Run on a single CPU
#SBATCH --time=23:50:00 # Time limit hrs:min:sec
#SBATCH --output=test_job%j.out # Standard output and error log
#SBATCH --gres=gpu:1
#SBATCH --partition=cl1_48h-1G
python3 main.py --epochs 400 --lr .00008 --num_layers 3 --num_mlp_layers 2--hidden_dim 400 --final_dropout .5 --graph_pooling_type average --neighbor_pooling_type sum --learn_eps --configfile R52
