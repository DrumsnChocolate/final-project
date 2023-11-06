#!/bin/bash
#for d in "cbis 2 prompt 224 64" "cbis 2 prompt 384 16" "cbis 2 prompt 672 4" "cbis 2 prompt 896 2"; do
for d in "cbis 2 finetune 224 1"; do
  set -- $d
  dataset=$1
  num_classes=$2
  transfer_type=$3
  crop_size=$4
  batch_size=$5
  if [ ${transfer_type} == "finetune" ]; then
    CUDA_VISIBLE_DEVICES=0,1,2,3 sbatch -J "vpt-vit-mammo-${dataset}" --constraint=rtx-6000 slurm/visual_prompt_tuning/vit_mammo.sbatch ${dataset} ${num_classes} ${transfer_type} ${crop_size} ${batch_size}
  else
    sbatch -J "vpt-vit-mammo-${dataset}" --gres=gpu:ampere:1 slurm/visual_prompt_tuning/vit_mammo.sbatch ${dataset} ${num_classes} ${transfer_type} ${crop_size} ${batch_size}
  fi
done