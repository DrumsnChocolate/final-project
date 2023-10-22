#!/bin/bash
for d in "cbis 2 finetune"; do
  set -- $d
  dataset=$1
  num_classes=$2
  transfer_type=$3
  if [ ${transfer_type} == "finetune" ]; then
    sbatch --gres=gpu:lovelace:1 --constraint=rtx-6000 -J "vpt-vit-mammo-${dataset}" slurm/visual_prompt_tuning/vit_mammo.sbatch ${dataset} ${num_classes} ${transfer_type}
  else
    sbatch -J "vpt-vit-mammo-${dataset}" slurm/visual_prompt_tuning/vit_mammo.sbatch ${dataset} ${num_classes} ${transfer_type}
  fi
done