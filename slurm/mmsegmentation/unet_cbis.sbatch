#!/bin/bash
#SBATCH -J unet
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --constraint=a40

lr=$1  # learning rate, float
wd=$2  # weight decay, float

config="configs/unet/unet-s5-d16_fcn_4xb4-160k_cityscapes_cbis-ddsm-binary-512x1024.py"
optim_options="optimizer.lr=$lr optimizer.weight_decay=$wd optim_wrapper.optimizer.lr=$lr optim_wrapper.optimizer.weight_decay=$wd"
normalization_options="data_preprocessor.mean='[68.882, 68.882, 68.882]' data_preprocessor.std='[66.631, 66.631, 66.631]' model.data_preprocessor.mean='[68.882, 68.882, 68.882]' model.data_preprocessor.std='[66.631, 66.631, 66.631]'"

cd implementation/mmsegmentation
source ~/miniconda3/etc/profile.d/conda.sh
conda activate openmmlab

python tools/train.py $config \
  --cfg-options \
  train_dataloader.batch_size=4 \
  train_dataloader.num_workers=4 \
  $optim_options \
  $normalization_options