#!/bin/bash
#SBATCH --gres=gpu:v100-32:1
#SBATCH -c 8
#SBATCH -o train_%j.out
set -x

../tools/train_net.py --config-file ../configs/COCO-Detection/retinanet_R_50_FPN_1x.yaml --num-gpus 1 SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.001 OUTPUT_DIR . TEST.EVAL_PERIOD 10 
