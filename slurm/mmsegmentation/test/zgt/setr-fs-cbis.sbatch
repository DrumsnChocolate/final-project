#!/bin/bash
#SBATCH -J zgt_setr_fs_cbis
#SBATCH -c 2
#SBATCH --mail-type=START,END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40
#SBATCH --time=0-01:00:00

# evaluate the From Scratch CBIS SETR on the zgt dataset
cd implementation/mmsegmentation
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mmsegmentation

timestamp="20240417_224050"
work_dir="work_dirs/setr_vit-l_pup-noaux_8xb2-220k_cbis-ddsm-mono-256x256/${timestamp}"

python tools/test.py ${work_dir} --cfg-options test_dataloader.dataset.data_root='data/zgt/zgt-linked' test_dataloader.dataset.data_prefix="dict(img_path='images',seg_map_path='annotations_binary')"

conda deactivate