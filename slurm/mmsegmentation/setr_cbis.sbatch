#!/bin/bash
#SBATCH -J setr
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --constraint=a40

lr=$1
wd=$2
transfer=$3
task=$4


cd implementation/mmsegmentation
source ~/miniconda3/etc/profile.d/conda.sh
conda activate openmmlab

if [ ${transfer} == "vpt" ]; then
  if [ ${task} == "cbis-binary" ]; then
    config="configs/setr_vpt/setrvpt_vit-l_pup_8xb2-160k_cbis-ddsm-binary-512x512.py"
  fi
  if [ ${task} == "cbis-multi" ]; then
    config="configs/setr_vpt/setrvpt_vit-l_pup_8xb2-160k_cbis-ddsm-multi-512x512.py"
  fi
fi

if [ ${transfer} == "full" ]; then
  if [ ${task} == "cbis-binary" ]; then
    config="configs/setr/setr_vit-l_pup_8xb2-160k_cbis-ddsm-binary-512x512.py"
  fi
  if [ ${task} == "cbis-multi" ]; then
    config="configs/setr/setr_vit-l_pup_8xb2-160k_cbis-ddsm-multi-512x512.py"
  fi
fi

class_weights=1,1
if [ ${task} == "cbis-binary" ]; then
  class_weights=0.04,1.96
fi

python tools/train.py $config \
  --cfg-options \
  train_dataloader.batch_size=4 \
  train_dataloader.num_workers=4 \
  optimizer.lr=$lr \
  optimizer.weight_decay=$wd \
  optim_wrapper.optimizer.lr=$lr \
  optim_wrapper.optimizer.weight_decay=$wd \
  record_gpu_snapshot=True \
  model.decode_head.loss_decode.class_weight=$class_weights

conda deactivate
