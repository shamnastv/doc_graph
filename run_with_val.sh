#!/bin/sh
#SBATCH --job-name=GNN # Job name
#SBATCH --ntasks=4 # Run on a single CPU
#SBATCH --time=23:50:00 # Time limit hrs:min:sec
#SBATCH --output=test_job%j.out # Standard output and error log
#SBATCH --gres=gpu:1
#SBATCH --partition=cl2_48h-1G
python3 train_with_val.py --batch_size 32 --iters_per_epoch 120 --update_freq 1000 --epochs 2000 --lr .00001 --lr_c .01 --num_mlp_layers 3 --num_mlp_layers_c 3 --hidden_dim 768 --final_dropout .5 --graph_pooling_type average --neighbor_pooling_type average --learn_eps --configfile config/R52.yaml --alpha 1500 --beta 100
