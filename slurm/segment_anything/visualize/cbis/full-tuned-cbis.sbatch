#!/bin/bash
#SBATCH -J cbis_sam_ft_vis
#SBATCH -c 2
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40
#SBATCH --time=0-01:00:00

module load nvidia/cuda-11.8
cd implementation/segment_anything
source ~/miniconda3/etc/profile.d/conda.sh
conda activate segment_anything

image_names=()  # todo: fill in image names
timestamp="20240416_104218"  # yyyymmdd_hhmmss

python finetune/visualize.py outputs/${timestamp}/config.yaml \
  --image_names "${image_names[@]}" \
  --output_dir visualization/sam_ft/cbis

conda deactivate
