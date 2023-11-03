#!/bin/bash
#SBATCH -J vpt-setr-cbis
#SBATCH -c 4
#SBATCH --gres=gpu:ampere:4
#SBATCH --mail-type=END,FAIL
gpu_num=4

cd implementation/mmsegmentation
source ~/miniconda3/etc/profile.d/conda.sh
conda activate openmmlab

tools/dist_train.sh configs/setr_vpt/setrvpt_vit-l_pup_8xb2-160k_cbis-ddsm-binary-512x512.py $gpu_num --cfg-options train_dataloader.batch_size=4