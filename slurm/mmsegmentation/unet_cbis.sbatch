#!/bin/bash
#SBATCH -J unet
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --constraint=a40

lr=$1  # learning rate, float
size=$2  # size of the image, 256 or 512
loss=$3  # loss function, "dice & iou" or "cross_entropy"


if [ $size == 256 ]; then
  config="configs/unet/unet-s5-d16_fcn-noaux_4xb4-220k_cbis-ddsm-mono-256x256.py"
elif [ $size == 512 ]; then
  config="configs/unet/unet-s5-d16_fcn-noaux_4xb4-220k_cbis-ddsm-mono-512x512.py"
else
  echo "Error: Unknown size"
  exit 1
fi

if [ $loss == "dice & iou" ]; then
  model_options=(model.decode_head.loss_decode="[dict(loss_weight=0.4,type='DiceLoss',naive_dice=True),dict(loss_weight=0.6,type='IoULoss')]")
elif [ $loss == "cross_entropy" ]; then
  model_options=(model.decode_head.loss_decode="dict(loss_weight=1.0,type='CrossEntropyLoss',use_sigmoid=true)")
else
  echo "Error: Unknown loss"
  exit 1
fi
model_options=(model.decode_head.loss_decode="[dict(loss_weight=0.4,type='DiceLoss',naive_dice=True),dict(loss_weight=0.6,type='IoULoss')]")
optim_options=(optimizer="dict(lr=${lr},type='Adam')" optim_wrapper.optimizer="dict(lr=${lr},type='Adam')" param_scheduler="[dict(type='ReduceOnPlateauLR',rule='greater',monitor='mIoU')]")
normalization_options=(data_preprocessor.mean='[68.882,68.882,68.882]' data_preprocessor.std='[66.631,66.631,66.631]' model.data_preprocessor.mean='[68.882,68.882,68.882]' model.data_preprocessor.std='[66.631,66.631,66.631]')

mkdir /local/${SLURM_JOBID}
## copy data to node
cd ${SLURM_SUBMIT_DIR}/implementation/mmsegmentation
cp -rL --parents data/cbis /local/${SLURM_JOBID}

## execute the python we want
source ~/miniconda3/etc/profile.d/conda.sh
conda activate openmmlab

python tools/train.py $config --data-root-prefix /local/${SLURM_JOBID} \
  --cfg-options \
  "${model_options[@]}" \
  "${optim_options[@]}" \
  "${normalization_options[@]}"


## clean up

rm -rf /local/${SLURM_JOBID}