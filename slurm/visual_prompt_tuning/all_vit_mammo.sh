#!/bin/bash
for d in "cbis 2"; do
  set -- $d
  dataset=$1
  num_classes=$2
  sbatch -J "vpt-vit-mammo-${dataset}" slurm/visual_prompt_tuning/vit_mammo.sbatch ${dataset} ${num_classes}
done