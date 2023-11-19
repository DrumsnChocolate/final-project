#!/bin/bash
dataset="cbis"
num_classes=2
#for transfer_type in "finetune" "prompt"; do
for transfer_type in "finetune"; do
  for patience in 7 14 21; do
    for img_size in 500 800; do
      if [ $img_size == 200 ]; then
        batch_size=64
      elif [ $img_size == 500 ]; then
        batch_size=32
      elif [ $img_size == 800 ]; then
        batch_size=8
      fi
      if [ ${transfer_type} == "finetune" ]; then
        CUDA_VISIBLE_DEVICES=0 sbatch -J "full-vit-mammo-${dataset}" --constraint=a40 --gres=gpu:1 slurm/visual_prompt_tuning/vit_mammo.sbatch $dataset $num_classes $transfer_type $img_size $batch_size $patience
      else
        sbatch -J "vpt-vit-mammo-${dataset}" --gres=gpu:ampere:1 slurm/visual_prompt_tuning/vit_mammo.sbatch $dataset $num_classes $transfer_type $img_size $batch_size $patience
      fi
    done
  done
done

#
#for d in "cbis 2 prompt 224 64" "cbis 2 prompt 384 16" "cbis 2 prompt 672 4" "cbis 2 prompt 896 2"; do
#  for d in "cbis 2 finetune 224 64"; do
#    set -- $d
#    dataset=$1
#    num_classes=$2
#    transfer_type=$3
#    img_size=$4
#    batch_size=$5
#    if [ ${transfer_type} == "finetune" ]; then
#      CUDA_VISIBLE_DEVICES=0 sbatch -J "full-vit-mammo-${dataset}" --constraint=rtx-6000 slurm/visual_prompt_tuning/vit_mammo.sbatch ${dataset} ${num_classes} ${transfer_type} ${img_size} ${batch_size}
#    else
#      sbatch -J "vpt-vit-mammo-${dataset}" --gres=gpu:ampere:1 slurm/visual_prompt_tuning/vit_mammo.sbatch ${dataset} ${num_classes} ${transfer_type} ${img_size} ${batch_size}
#    fi
#  done
#done
