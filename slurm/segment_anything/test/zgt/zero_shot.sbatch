#!/bin/bash
#SBATCH -J zgt_zero_shot
#SBATCH -c 2
#SBATCH --mail-type=START,END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40

# evaluate the unmodified SAM on the zgt dataset
cd implementation/segment_anything
source ~/miniconda3/etc/profile.d/conda.sh
conda activate segment_anything

python finetune/zero_shot.py finetune/configs/sam_zs_zgt-binary.yaml

conda deactivate
