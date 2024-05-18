#!/bin/bash
#SBATCH -J cbis_setr-ft_vis
#SBATCH -c 2
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40
#SBATCH --time=0-01:00:00

# visualize the Full Tuned CBIS SETR on the cbis dataset
module load nvidia/cuda-11.8
cd implementation/mmsegmentation
source ~/miniconda3/etc/profile.d/conda.sh
conda activate openmmlab

timestamp="20240417_234236"
work_dir="work_dirs/setr_vit-l_pup-noaux_8xb2-220k_cbis-ddsm-mono-256x256/${timestamp}"

python tools/val.py ${work_dir} \
  --cfg-options \
  val_dataloader.sampler='dict(type="RandomSampler",replacement=False,num_samples=10,seed=218)' \
  default_hooks.visualization.interval=1 \
  default_hooks.visualization.draw=True \
  visualizer.alpha=1 \
  --save-dir visualization/setr_ft/cbis

conda deactivate