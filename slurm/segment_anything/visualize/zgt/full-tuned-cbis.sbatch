#!/bin/bash
#SBATCH -J zgt_sam_ft_vis
#SBATCH -c 2
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40
#SBATCH --time=0-01:00:00

module load nvidia/cuda-11.8
cd implementation/segment_anything
source ~/miniconda3/etc/profile.d/conda.sh
conda activate segment_anything

image_names=("1.2.826.0.1.3680043.2.526.11.40.16082154785849364.19568672.17414_processed.png" "1.2.826.0.1.3680043.2.526.11.40.16082154785849364.18458990.67703_processed.png" "1.2.826.0.1.3680043.2.526.11.40.16082154785849364.18733299.67623_processed.png" "1.2.826.0.1.3680043.2.526.11.40.16082154785849364.10888230.56970_processed.png" "1.2.826.0.1.3680043.2.526.11.40.1608721769240233.17816415.865759_processed.png" "1.2.826.0.1.3680043.2.526.11.40.16082154785849364.13096975.76555_processed.png" "1.2.826.0.1.3680043.2.526.11.40.1608721769240233.3052648.7915184_processed.png" "" "1.2.826.0.1.3680043.2.526.11.40.16082154785849364.16952853.43123_processed.png" "1.2.826.0.1.3680043.2.526.11.40.16082154785849364.10511845.73636_processed.png" "1.2.826.0.1.3680043.2.526.11.40.1608721769240233.13842130.708254_processed.png")
timestamp="20240416_104218"  # yyyymmdd_hhmmss

python finetune/visualize.py outputs/${timestamp}/config.yaml \
  --image_names "${image_names[@]}" \
  --output_dir visualization/sam_ft/zgt \
  --cfg-options \
  data.root="data/zgt/zgt-linked" \
  data.val=null \
  data.train=null \
  data.test="{annotation_dir: 'annotations_binary', batch_size: 1, image_dir: 'images'}"

conda deactivate
