#!/bin/bash
#SBATCH -J setr
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --constraint=a40

lr=0.0001
size=256
loss="cross_entropy"

if [ "$size" == "256" ]; then
  config="configs/setr/setr_vit-l_pup-noaux_8xb2-220k_cbis-ddsm-mono-256x256.py"
else
  echo "Error: Unknown size"
  exit 1
fi

param_scheduler_metric='mIoU'
param_scheduler_rule='greater'
if [ "$loss" == "cross_entropy" ]; then
  param_scheduler_metric='mCE'
  model_options=(model.decode_head.loss_decode="dict(loss_weight=1.0,type='CrossEntropyLoss',use_sigmoid=true)")
elif [ "$loss" == "iou" ]; then
  model_options=(model.decode_head.loss_decode="dict(loss_weight=1.0,type='IoULoss')")
elif [ "$loss" == "dice" ]; then
  model_options=(model.decode_head.loss_decode="dict(loss_weight=1.0,type='DiceLoss',naive_dice=true)")
else
  echo "Error: Unknown loss"
  exit 1
fi

if [ "$param_scheduler_metric" == 'mCE' ]; then
  param_scheduler_rule='less'
fi
stopping_metric="train_loss"
stopping_rule="less"
stopping_patience=40

optimizer="dict(lr=${lr},type='Adam')"
optim_options=(optimizer="$optimizer" optim_wrapper.optimizer="$optimizer" param_scheduler="[dict(type='ReduceOnPlateauLR',rule='${param_scheduler_rule}',monitor='${param_scheduler_metric}',min_value=0,cooldown=0,patience=25,threshold=0.0001,factor=0.1)]")
stopping_options=(custom_hooks="[dict(type='EarlyStoppingHook',monitor='${stopping_metric}',patience=${stopping_patience},rule='${stopping_rule}',min_delta=0)]")

normalization_options=(data_preprocessor.mean='[0.0,0.0,0.0]' data_preprocessor.std='[255,255,255]' model.data_preprocessor.mean='[0.0,0.0,0.0]' model.data_preprocessor.std='[255,255,255]')
augmentation_options=(train_dataloader.dataset.pipeline="[dict(type='LoadImageFromFile'),dict(type='LoadAnnotations',reduce_zero_label=True),dict(type='CLAHE'),dict(scale=($size,$size),type='Resize'),dict(type='PackSegInputs')]" train_pipeline="[dict(type='LoadImageFromFile'),dict(type='LoadAnnotations',reduce_zero_label=True),dict(type='CLAHE'),dict(scale=($size,$size),type='Resize'),dict(type='PackSegInputs')]")
pretrain_options=(model.backbone.init_cfg=None)

cd ${SLURM_SUBMIT_DIR}/implementation/mmsegmentation

source ~/miniconda3/etc/profile.d/conda.sh
conda activate openmmlab

python tools/train.py $config --cfg-options \
  "${model_options[@]}" \
  "${optim_options[@]}" \
  "${normalization_options[@]}" \
  "${augmentation_options[@]}" \
  "${stopping_options[@]}" \
  "${pretrain_options[@]}"
