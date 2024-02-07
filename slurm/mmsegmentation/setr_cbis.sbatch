#!/bin/bash
#SBATCH -J setr
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --constraint=a40

if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ] || [ -z "$5" ] || [ -z "$6" ] || [ -z "$7" ]; then
    echo "Error: Missing or empty command-line arguments."
    exit 1
fi


lr=$1  # learning rate, float
wd=$2  # weight decay, float
transfer=$3  # choice between vpt and full
task=$4  # choice between cbis-binary and cbis-multi
balanced=$5  # choice between true and false
optimizer=$6  # choice between adamw and sgd
auxiliary_head=$7  # choice between true and false


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

if [ ${balanced} == "true" ]; then
  if [ ${task} == "cbis-binary" ]; then
    class_weights=0.04,1.96 # two classes
  fi

  if [ ${task} == "cbis-multi" ]; then
    # I haven't figured out what to do in this case yet. exit with an exception
    echo "Error: balanced multi-class training not implemented yet"
    exit 1
  fi
else
  if [ ${task} == "cbis-binary" ]; then
    class_weights=1,1 # two classes
  fi

  if [ ${task} == "cbis-multi" ]; then
    class_weights=1,1,1,1,1  # five classes
  fi
fi


if [ ${optimizer} == "adamw" ]; then
  optim_options=(optimizer="dict(type='AdamW',lr=$lr,weight_decay=$wd)" optim_wrapper.optimizer="dict(type='AdamW',lr=$lr,weight_decay=$wd)")
elif [ ${optimizer} == "sgd" ]; then
  optim_options=(optimizer.lr=$lr optimizer.weight_decay=$wd optim_wrapper.optimizer.lr=$lr optim_wrapper.optimizer.weight_decay=$wd)
else
  echo "Error: optimizer must be adamw or sgd"
  exit 1
fi

if [ ${auxiliary_head} == "true" ]; then
  auxiliary_head_options="model.auxiliary_head.0.loss_decode.class_weight=$class_weights model.auxiliary_head.1.loss_decode.class_weight=$class_weights model.auxiliary_head.2.loss_decode.class_weight=$class_weights"
elif [ ${auxiliary_head} == "false" ]; then
  auxiliary_head_options="model.auxiliary_head=\"[]\""
else
  echo "Error: auxiliary_head must be true or false"
  exit 1
fi

normalization_options="data_preprocessor.mean='[68.882,68.882,68.882]' data_preprocessor.std='[66.631,66.631,66.631]' model.data_preprocessor.mean='[68.882,68.882,68.882]' model.data_preprocessor.std='[66.631,66.631,66.631]'"


cd implementation/mmsegmentation
source ~/miniconda3/etc/profile.d/conda.sh
conda activate openmmlab

python tools/train.py $config \
  --cfg-options \
  train_dataloader.batch_size=4 \
  train_dataloader.num_workers=4 \
  "${optim_options[@]}" \
  record_gpu_snapshot=True \
  model.decode_head.loss_decode.class_weight=$class_weights \
  $auxiliary_head_options \
  $normalization_options


conda deactivate
