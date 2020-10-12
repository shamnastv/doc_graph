#!/bin/sh
#SBATCH --job-name=GNN # Job name
#SBATCH --ntasks=4 # Run on a single CPU
#SBATCH --time=23:50:00 # Time limit hrs:min:sec
#SBATCH --output=test_job%j.out # Standard output and error log
#SBATCH --gres=gpu:1
#SBATCH --partition=cl1_48h-1G
python3 main_mod.py --epochs 200 --iters_per_epoch 100 --lr .00005 --lr_c .01 --num_layers 3 --num_mlp_layers 2 --num_mlp_layers_c 3 --hidden_dim 300 --final_dropout .5 --graph_pooling_type average --neighbor_pooling_type sum --learn_eps --configfile config/R8.yaml --alpha 1500
