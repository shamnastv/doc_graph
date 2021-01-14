#!/bin/sh
#SBATCH --job-name=R8 # Job name
#SBATCH --ntasks=4 # Run on a single CPU
#SBATCH --time=23:50:00 # Time limit hrs:min:sec
#SBATCH --output=test_job%j%x.out # Standard output and error log
#SBATCH --gres=gpu:1
#SBATCH --partition=cl1_48h-1G

python3 main.py --epochs 400 --lr .00005 --num_layers 3 --num_mlp_layers 1 --hidden_dim 200 --final_dropout .5 --configfile R8_save --num_heads 2
python3 main.py --epochs 400 --lr .00008 --num_layers 3 --num_mlp_layers 1 --hidden_dim 200 --final_dropout .5 --configfile R8_save --num_heads 2
python3 main.py --epochs 400 --lr .0001 --num_layers 3 --num_mlp_layers 1 --hidden_dim 200 --final_dropout .5 --configfile R8_save --num_heads 2
python3 main.py --epochs 400 --lr .0003 --num_layers 3 --num_mlp_layers 1 --hidden_dim 200 --final_dropout .5 --configfile R8_save --num_heads 2
