#!/bin/bash
#SBATCH -J setr
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --constraint=a40

lr="$1"
size="$2"
batch_size="$3"
pretrained="$4"
loss="$5"
tuning_method="$6"
if [ "$tuning_method" == "full" ]; then
  config="configs/setr/setr_vit-l_pup-noaux_8xb2-220k_cbis-ddsm-mono-256x256.py"
elif [ "$tuning_method" == "vpt" ]; then
  config="configs/setr_vpt/setrvpt_vit-l_pup-noaux_8xb2-220k_cbis-ddsm-mono-256x256.py"
else
  echo "Error: Unknown tuning method"
  exit 1
fi

sample_count=2212
val_interval=$(($sample_count / $batch_size))
batch_settings=( train_dataloader.batch_size="$batch_size" train_cfg.val_interval="$val_interval" )



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

optimizer="dict(lr=${lr},type='AdamW')"
optim_options=(optimizer="$optimizer" optim_wrapper.optimizer="$optimizer" param_scheduler="[dict(type='ReduceOnPlateauLR',rule='${param_scheduler_rule}',monitor='${param_scheduler_metric}',min_value=0,cooldown=0,patience=25,threshold=0.0001,factor=0.1)]")
stopping_options=(custom_hooks="[dict(type='EarlyStoppingHook',monitor='${stopping_metric}',patience=${stopping_patience},rule='${stopping_rule}',min_delta=0)]")

normalization_options=(data_preprocessor.mean='[0.0,0.0,0.0]' data_preprocessor.std='[255,255,255]' model.data_preprocessor.mean='[0.0,0.0,0.0]' model.data_preprocessor.std='[255,255,255]')
train_pipeline="[dict(type='LoadImageFromFile'),dict(type='LoadAnnotations',reduce_zero_label=True),dict(type='CLAHE'),dict(scale=($size,$size),type='Resize'),dict(type='PackSegInputs')]"
augmentation_options=(train_dataloader.dataset.pipeline="$train_pipeline" train_pipeline="$train_pipeline")

# further options related to size

test_size_options=(test_dataloader.dataset.pipeline.2.scale="($size,$size)" test_pipeline.2.scale="($size,$size)")
val_size_options=(val_dataloader.dataset.pipeline.2.scale="($size,$size)" val_pipeline.2.scale="($size,$size)")
overall_size_options=(crop_size="($size,$size)" data_preprocessor.size="($size,$size)" data_preprocessor.test_cfg.size="($size,$size)" model.data_preprocessor.size="($size,$size)" model.data_preprocessor.test_cfg.size="($size,$size)" model.backbone.img_size="($size,$size)")

size_options=("${test_size_options[@]}" "${val_size_options[@]}" "${overall_size_options[@]}")

if [ "$pretrained" == "true" ]; then
  if [ "$size" == "224" ]; then
    pretrain_options=(model.backbone.init_cfg.checkpoint="pretrain/vit_large_p16_224.pth")
  elif [ "$size" == "384" ]; then
    pretrain_options=(model.backbone.init_cfg.checkpoint="pretrain/vit_large_p16.pth")
  else
    echo "Error: Unknown size"
    exit 1
  fi
else
  pretrain_options=(model.backbone.init_cfg=None)
fi

cd ${SLURM_SUBMIT_DIR}/implementation/mmsegmentation

source ~/miniconda3/etc/profile.d/conda.sh
conda activate openmmlab

python tools/train.py $config --cfg-options \
  "${model_options[@]}" \
  "${optim_options[@]}" \
  "${normalization_options[@]}" \
  "${augmentation_options[@]}" \
  "${stopping_options[@]}" \
  "${pretrain_options[@]}" \
  "${size_options[@]}" \
  "${batch_settings[@]}"
