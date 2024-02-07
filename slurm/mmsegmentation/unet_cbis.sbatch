#!/bin/bash
#SBATCH -J unet
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --constraint=a40

lr=$1  # learning rate, float

config="configs/unet/unet-s5-d16_fcn-noaux_4xb4-220k_cbis-ddsm-mono-256x256.py"
model_options=(model.decode_head.loss_decode="dict(loss_weight=1,type='DiceLoss',naive_dice=True)")
#optim_options="optimizer.lr=$lr optimizer.weight_decay=$wd optim_wrapper.optimizer.lr=$lr optim_wrapper.optimizer.weight_decay=$wd"
optim_options=(optimizer="dict(lr=${lr},type='Adam')" optim_wrapper.optimizer="dict(lr=${lr},type='Adam')" param_scheduler=None)
normalization_options=(data_preprocessor.mean='[68.882,68.882,68.882]' data_preprocessor.std='[66.631,66.631,66.631]' model.data_preprocessor.mean='[68.882,68.882,68.882]' model.data_preprocessor.std='[66.631,66.631,66.631]')

cd implementation/mmsegmentation
source ~/miniconda3/etc/profile.d/conda.sh
conda activate openmmlab

python tools/train.py $config \
  --cfg-options \
  "${model_options[@]}" \
  "${optim_options[@]}" \
  "${normalization_options[@]}"