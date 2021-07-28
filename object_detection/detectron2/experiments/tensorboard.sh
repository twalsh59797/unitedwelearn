#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH -o tensorboard_%j.out --time=24:00:00
set -x

tensorboard --logdir=coco_retinanet_baseline --bind_all --port=6030

