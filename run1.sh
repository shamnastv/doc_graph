#!/bin/sh
#SBATCH --job-name=GNN # Job name
#SBATCH --ntasks=4 # Run on a single CPU
#SBATCH --time=23:50:00 # Time limit hrs:min:sec
#SBATCH --output=test_job%j.out # Standard output and error log
#SBATCH --gres=gpu:1
#SBATCH --partition=cl1_48h-1G
python3 main_mod.py --epochs 200 --lr .00001 --num_layers 3 --num_mlp_layers 2 --hidden_dim 768 --final_dropout .5 --graph_pooling_type average --neighbor_pooling_type sum --learn_eps --configfile param.yaml --beta .1
