#!/bin/bash
#SBATCH -c 2
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40

cd implementation/segment_anything
source ~/miniconda3/etc/profile.d/conda.sh
conda activate segment_anything
#python finetune/train.py finetune/configs/sam_full_cbis-binary.yaml
python finetune/train.py finetune/configs/sam_full_cbis-binary.yaml --cfg-options model.loss.parts="[{'name': 'Dice', 'invert': True, 'weight': 1}]"
conda deactivate