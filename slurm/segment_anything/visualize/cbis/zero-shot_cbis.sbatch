#!/bin/bash
#SBATCH -J cbis_sam_zs_vis
#SBATCH -c 2
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40
#SBATCH --time=0-01:00:00

module load nvidia/cuda-11.8
cd implementation/segment_anything
source ~/miniconda3/etc/profile.d/conda.sh
conda activate segment_anything

image_names=("Mass-Training_P_00559_LEFT_MLO_1-1.png_0.png" "Calc-Training_P_00312_RIGHT_MLO_1-1.png_0.png" "Calc-Training_P_01847_LEFT_MLO_1-1.png_0.png" "Calc-Training_P_01375_LEFT_MLO_1-1.png_0.png" "Mass-Training_P_00264_LEFT_CC_1-1.png_0.png" "Mass-Training_P_00128_LEFT_CC_1-1.png_0.png" "Calc-Training_P_01156_LEFT_CC_1-1.png_0.png" "Calc-Training_P_01545_RIGHT_MLO_1-1.png_0.png" "" "Mass-Training_P_00801_LEFT_MLO_1-1.png_0.png" "Calc-Training_P_00667_RIGHT_CC_1-1.png_0.png")

python finetune/visualize.py finetune/configs/sam_full_cbis-binary.yaml \
  --image_names "${image_names[@]}" \
  --output_dir visualization/sam_zs/cbis

conda deactivate
