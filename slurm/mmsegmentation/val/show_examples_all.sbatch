#!/bin/bash
#SBATCH -J show-examples-setr-all
#SBATCH -c 8
#SBATCH --gres=gpu:ampere:1
#SBATCH --mail-type=END,FAIL

cd implementation/mmsegmentation
source ~/miniconda3/etc/profile.d/conda.sh
conda activate openmmlab

# list all the setr runs so far, by matching work_dirs/*/202*/last_checkpoint.pth

for i in $(find work_dirs/*/202* -name last_checkpoint); do
  # obtain directory name from path
  dir=$(dirname $i)
  python tools/val.py $dir --cfg-options val_dataloader.sampler='dict(type="RandomSampler",replacement=False,num_samples=20,seed=218)' default_hooks.visualization.interval=1 default_hooks.visualization.draw=True visualizer.alpha=0.5
done