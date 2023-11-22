#!/bin/bash
#SBATCH -J vpt-setr-cbis
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --constraint=a40

lr=$1
wd=$2


cd implementation/mmsegmentation
source ~/miniconda3/etc/profile.d/conda.sh
conda activate openmmlab

python tools/train.py configs/setr_vpt/setrvpt_vit-l_pup_8xb2-160k_cbis-ddsm-binary-512x512.py \
  --cfg-options \
  train_dataloader.batch_size=4 \
  optimizer.lr=$lr \
  optimizer.weight_decay=$wd \
  optim_wrapper.optimizer.lr=$lr \
  optim_wrapper.optimizer.weight_decay=$wd

conda deactivate