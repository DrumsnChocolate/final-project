model_root=visual_prompt_tuning/model_root
data_path=visual_prompt_tuning/data_path
output_dir=visual_prompt_tuning/output_dir

dataset=$1
num_classes=$2
transfer_type=$3
crop_size=$4
batch_size=$5

config=visual_prompt_tuning/configs/prompt/cub.yaml
if [ ${transfer_type} == "finetune" ]; then
  config=visual_prompt_tuning/configs/finetune/cub.yaml
fi

python visual_prompt_tuning/tune_cbis.py \
  --config-file ${config} \
  --train-type ${transfer_type} \
  MODEL.TYPE "vit" \
  DATA.BATCH_SIZE ${batch_size} \
  MODEL.PROMPT.NUM_TOKENS "50" \
  MODEL.PROMPT.DEEP "True" \
  MODEL.PROMPT.DROPOUT "0.1" \
  DATA.FEATURE "sup_vitb16_imagenet21k" \
  DATA.NAME "mammo-${dataset}" \
  DATA.NUMBER_CLASSES "${num_classes}" \
  DATA.CROPSIZE ${crop_size} \
  MODEL.MODEL_ROOT "${model_root}" \
  DATA.DATAPATH "${data_path}" \
  OUTPUT_DIR "${output_dir}" \
  SOLVER.PATIENCE "7" \
  SOLVER.CRITERION "loss"