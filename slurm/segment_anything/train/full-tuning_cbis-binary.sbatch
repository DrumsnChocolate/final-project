#!/bin/bash
#SBATCH --job-name=ft_sam_cbis
#SBATCH -c 2
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40

cd implementation/segment_anything
source ~/miniconda3/etc/profile.d/conda.sh
conda activate segment_anything
python finetune/train.py finetune/configs/sam_full_cbis-binary.yaml --cfg-options model.pixel_mean="[68.882,68.882,68.882]" model.pixel_std="[66.631,66.631,66.631]"
conda deactivate
