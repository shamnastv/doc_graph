#!/bin/sh
#SBATCH --job-name=GNN # Job name
#SBATCH --ntasks=4 # Run on a single CPU
#SBATCH --time=23:50:00 # Time limit hrs:min:sec
#SBATCH --output=test_job%j.out # Standard output and error log
#SBATCH --gres=gpu:1
#SBATCH --partition=q2h_24h-2G
python3 main.py --iters_per_epoch 3 --epochs 400 --lr .00001 --lr_c .01 --num_mlp_layers 4 --num_mlp_layers_c 3 --hidden_dim 768 --final_dropout .5 --graph_pooling_type average --neighbor_pooling_type average --learn_eps --configfile config/R8.yaml --alpha .1
