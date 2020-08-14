#!/bin/sh
#SBATCH --job-name=GNN # Job name
#SBATCH --ntasks=1 # Run on a single CPU
#SBATCH --time=11:50:00 # Time limit hrs:min:sec
#SBATCH --output=test_job%j.out # Standard output and error log
#SBATCH --gres=gpu:1
#SBATCH --partition=q2h_12h-32C
python3 main.py --batch_size 64 --iters_per_epoch 50 --epochs 800 --lr .0001 --num_layers 6 --num_mlp_layers 3 --hidden_dim 768 --final_dropout .5 --graph_pooling_type average --neighbor_pooling_type average --learn_eps --configfile param.yaml