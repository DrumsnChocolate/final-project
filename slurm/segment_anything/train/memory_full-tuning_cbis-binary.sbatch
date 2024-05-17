#!/bin/bash
#SBATCH --job-name=mem_sam_ft
#SBATCH -c 2
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40
#SBATCH --time=0-01:00:00

cd implementation/segment_anything
source ~/miniconda3/etc/profile.d/conda.sh
conda activate segment_anything
python finetune/train.py finetune/configs/sam_full_cbis-binary.yaml --cfg-options schedule.epochs=1
conda deactivate
