#!/bin/bash
#SBATCH --gres=gpu:v100-32:1
#SBATCH -c 8
#SBATCH -o tide_%j.out
set -x

python tide.py 
