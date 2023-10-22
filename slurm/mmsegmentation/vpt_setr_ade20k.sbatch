#!/bin/bash
#SBATCH -J vpt-setr-ade20k
#SBATCH -c 4
#SBATCH --gres=gpu:ampere:4
#SBATCH --mail-type=END,FAIL

cd implementation/mmsegmentation
source ~/miniconda3/etc/profile.d/conda.sh
conda activate openmmlab

python tools/train.py configs/setr_vpt/setrvpt_vit-l_pup_8xb2-160k_ade20k-512x512.py --cfg-options train_dataloader.batch_size=4

conda deactivate
