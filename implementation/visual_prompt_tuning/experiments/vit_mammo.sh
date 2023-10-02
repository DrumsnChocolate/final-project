model_root=visual_prompt_tuning/model_root
data_path=visual_prompt_tuning/data_path
output_dir=visual_prompt_tuning/output_dir

dataset=$1
num_classes=$2

python visual_prompt_tuning/tune_cbis.py \
  --config-file visual_prompt_tuning/configs/prompt/cub.yaml \
  --train-type "prompt" \
  MODEL.TYPE "vit" \
  DATA.BATCH_SIZE "64" \
  MODEL.PROMPT.NUM_TOKENS "50" \
  MODEL.PROMPT.DEEP "True" \
  MODEL.PROMPT.DROPOUT "0.1" \
  DATA.FEATURE "sup_vitb16_imagenet21k" \
  DATA.NAME "mammo-${dataset}" \
  DATA.NUMBER_CLASSES "${num_classes}" \
  DATA.CROPSIZE "224" \
  MODEL.MODEL_ROOT "${model_root}" \
  DATA.DATAPATH "${data_path}" \
  OUTPUT_DIR "${output_dir}" \
  SOLVER.PATIENCE "7" \
  SOLVER.CRITERION "loss"