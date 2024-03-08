#!/bin/bash
#SBATCH -J unet
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --constraint=a40
#SBATCH --time=4-00:00:00

lr=$1  # learning rate, float
size=$2  # size of the image, 256 only option right now
loss=$3  # loss function, "dice & iou" or "cross_entropy"
clahe=$4  # boolean; whether we use clahe and therefore need to standardize to [0,1], or standardize with z-score
augmentation=$5  # boolean; whether we use any data augmentation besides CLAHE or not.
mass_only=$6  # boolean; whether we only use the mass data or not.



if [ "$size" == "256" ]; then
  config="configs/unet/unet-s5-d16_fcn-noaux_4xb4-220k_cbis-ddsm-mono-256x256.py"
else
  echo "Error: Unknown size"
  exit 1
fi


param_scheduler_metric='mIoU'
param_scheduler_rule='greater'
if [ "$loss" == "dice & iou" ]; then
  model_options=(model.decode_head.loss_decode="[dict(loss_weight=0.4,type='DiceLoss',naive_dice=True),dict(loss_weight=0.6,type='IoULoss')]")
elif [ "$loss" == "cross_entropy" ]; then
  param_scheduler_metric='mCE'
  model_options=(model.decode_head.loss_decode="dict(loss_weight=1.0,type='CrossEntropyLoss',use_sigmoid=true)")
elif [ "$loss" == "cross_entropy & iou" ]; then
  model_options=(model.decode_head.loss_decode="[dict(loss_weight=0.5,type='CrossEntropyLoss',use_sigmoid=true),dict(loss_weight=0.5,type='IoULoss')]")
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

optim_options=(optimizer="dict(lr=${lr},type='Adam')" optim_wrapper.optimizer="dict(lr=${lr},type='Adam')" param_scheduler="[dict(type='ReduceOnPlateauLR',rule='${param_scheduler_rule}',monitor='${param_scheduler_metric}',min_value=0,cooldown=0,patience=25,threshold=0.0001,factor=0.1)]")
stopping_options=(custom_hooks="[dict(type='EarlyStoppingHook',monitor='${stopping_metric}',patience=${stopping_patience},rule='${stopping_rule}',min_delta=0)]")

if [ "$clahe" == "true" ]; then
  normalization_options=(data_preprocessor.mean='[0.0,0.0,0.0]' data_preprocessor.std='[255,255,255]' model.data_preprocessor.mean='[0.0,0.0,0.0]' model.data_preprocessor.std='[255,255,255]')
else
  normalization_options=(data_preprocessor.mean='[68.882,68.882,68.882]' data_preprocessor.std='[66.631,66.631,66.631]' model.data_preprocessor.mean='[68.882,68.882,68.882]' model.data_preprocessor.std='[66.631,66.631,66.631]')
fi

if [ "$augmentation" == "false" ]; then
  augmentation_options=(train_dataloader.dataset.pipeline="[dict(type='LoadImageFromFile'),dict(type='LoadAnnotations',reduce_zero_label=True),dict(type='CLAHE'),dict(scale=($size,$size),type='Resize'),dict(type='PackSegInputs')]" train_pipeline="[dict(type='LoadImageFromFile'),dict(type='LoadAnnotations',reduce_zero_label=True),dict(type='CLAHE'),dict(scale=($size,$size),type='Resize'),dict(type='PackSegInputs')]")
else
  augmentation_options=()
fi

data_source_options=()
schedule_options=()
if [ "$mass_only" == "true" ]; then
  train_loader_source_options=(train_dataloader.dataset.data_prefix="dict(img_path='images_mass/train',seg_map_path='annotations_mass_binary/train')")
  val_loader_source_options=(val_dataloader.dataset.data_prefix="dict(img_path='images_mass/val',seg_map_path='annotations_mass_binary/val')")
  test_loader_source_options=(test_dataloader.dataset.data_prefix="dict(img_path='images_mass/test',seg_map_path='annotations_mass_binary/test')")
  data_source_options=("${train_loader_source_options[@]}" "${val_loader_source_options[@]}" "${test_loader_source_options[@]}")
  schedule_options=(train_cfg.val_interval=69) # roughly 1106 samples, divided by batch size 16
fi

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
  "${normalization_options[@]}" \
  "${augmentation_options[@]}" \
  "${stopping_options[@]}" \
  "${data_source_options[@]}" \
  "${schedule_options[@]}"


## clean up

rm -rf /local/${SLURM_JOBID}
