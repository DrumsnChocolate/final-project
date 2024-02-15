#!/bin/bash
#SBATCH -J show-examples-setr-all
#SBATCH -c 8
#SBATCH --gres=gpu:ampere:1
#SBATCH --mail-type=END,FAIL

cd implementation/mmsegmentation
source ~/miniconda3/etc/profile.d/conda.sh
conda activate openmmlab

# list all the setr runs so far, by matching work_dirs/*/202*/last_checkpoint.pth

for i in $(find work_dirs/unet*/202* -name last_checkpoint); do
  # obtain directory name from path
  dir=$(dirname $i)
  # run the validation script only when there is not yet a folder that looks like $dir/val/202*/vis_data/vis_image
  shopt -s nullglob  # simply return nothing when glob is not matched
  vis_folders=( $dir/val/202*/vis_data/vis_image )  # get a list of matching folders
  if ! (( ${#vis_folders[@]} )); then  # if there not a single in vis_folders, then
    python tools/val.py $dir --cfg-options val_dataloader.sampler='dict(type="RandomSampler",replacement=False,num_samples=20,seed=218)' default_hooks.visualization.interval=1 default_hooks.visualization.draw=True visualizer.alpha=0.5
  fi
done