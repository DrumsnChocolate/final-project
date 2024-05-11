#!/bin/bash
#SBATCH -J zgt_unet_fs_cbis
#SBATCH -c 2
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40
#SBATCH --time=0-01:00:00

# evaluate the From Scratch CBIS U-Net on the zgt dataset
module load nvidia/cuda-11.8
cd implementation/mmsegmentation
source ~/miniconda3/etc/profile.d/conda.sh
conda activate openmmlab

timestamp="20240417_223739"
work_dir="work_dirs/unet-s5-d16_fcn-noaux_4xb4-220k_cbis-ddsm-mono-256x256/${timestamp}"

python tools/test.py ${work_dir} --cfg-options test_dataloader.dataset.data_root='data/zgt/zgt-linked' test_dataloader.dataset.data_prefix="dict(img_path='images',seg_map_path='annotations_binary')"

conda deactivate