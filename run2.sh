#!/bin/sh
#SBATCH --job-name=GNN # Job name
#SBATCH --ntasks=1 # Run on a single CPU
#SBATCH --time=11:50:00 # Time limit hrs:min:sec
#SBATCH --output=test_job%j.out # Standard output and error log
#SBATCH --gres=gpu:5
#SBATCH --partition=q2h_12h-2G
python3 main1.py --epochs 1000 --lr .00001 --num_mlp_layers 4 --hidden_dim 768 --final_dropout .5 --graph_pooling_type average --neighbor_pooling_type average --learn_eps --configfile param.yaml --alpha 100
