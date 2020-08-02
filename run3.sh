#!/bin/sh
#SBATCH --job-name=GNN # Job name
#SBATCH --ntasks=1 # Run on a single CPU
#SBATCH --time=11:50:00 # Time limit hrs:min:sec
#SBATCH --output=test_job%j.out # Standard output and error log
#SBATCH --gres=gpu:1
#SBATCH --partition=q2h_12h-32C
python3 gin_main.py --batch_size 64 --iters_per_epoch 50 --epochs 700 --lr .001 --num_layers 5 --num_mlp_layers 2 --hidden_dim 769 --final_dropout .5 --graph_pooling_type average --neighbor_pooling_type average