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

image_names=()  # todo: fill in image names

python finetune/visualize.py finetune/configs/sam_full_cbis-binary.yaml \
  --image_names "${image_names[@]}" \
  --output_dir visualization/sam_zs/zgt \
  --cfg-options \
  data.root="data/zgt/zgt-linked" \
  data.val=null \
  data.train=null \
  data.test="{annotation_dir: 'annotations_binary', batch_size: 1, image_dir: 'images'}"


conda deactivate
