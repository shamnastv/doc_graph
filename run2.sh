#!/bin/sh
#SBATCH --job-name=GNN # Job name
#SBATCH --ntasks=4 # Run on a single CPU
#SBATCH --time=23:50:00 # Time limit hrs:min:sec
#SBATCH --output=test_job%j.out # Standard output and error log
#SBATCH --gres=gpu:1
#SBATCH --partition=q2h_24h-2G
python3 main_mod.py --device 0 --epochs 1000 --iters_per_epoch 100 --lr .0001 --lr_c .05 --num_layers 5 --num_mlp_layers 3 --num_mlp_layers_c 3 --hidden_dim 768 --final_dropout .5 --graph_pooling_type average --neighbor_pooling_type sum --configfile param.yaml --alpha .1
