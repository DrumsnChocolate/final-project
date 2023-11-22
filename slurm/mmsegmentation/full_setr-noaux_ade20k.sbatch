#!/bin/bash
#SBATCH -J full-setr-noaux-ade20k
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --constraint=a40

cd implementation/mmsegmentation
source ~/miniconda3/etc/profile.d/conda.sh
conda activate openmmlab

python tools/train.py configs/setr/setr_vit-l_pup-noaux_8xb2-160k_ade20k-512x512.py --cfg-options train_dataloader.batch_size=8 train_dataloader.num_workers=8 record_gpu_snapshot=True

conda deactivate
