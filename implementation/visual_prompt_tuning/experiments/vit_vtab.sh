model_root=visual_prompt_tuning/model_root
data_path=visual_prompt_tuning/data_path
output_dir=visual_prompt_tuning/output_dir

dataset=$1
num_classes=$2

## parameters are specified based on the visual prompt tuning paper, and the hyper parameter tuning that was done there
#for seed in "42" "44" "82" "100" "800"; do
#  python visual_prompt_tuning/train.py \
#    --config-file visual_prompt_tuning/configs/prompt/cub.yaml \
#    MODEL.TYPE "vit" \
#    DATA.BATCH_SIZE "64" \
#    MODEL.PROMPT.NUM_TOKENS "100" \
#    MODEL.PROMPT.DEEP "True" \
#    MODEL.PROMPT.DROPOUT "0.1" \
#    DATA.FEATURE "sup_vitb16_imagenet21k" \
#    DATA.NAME "vtab-${dataset}" \
#    DATA.NUMBER_CLASSES "${num_classes}" \
#    SOLVER.BASE_LR "5.0" \
#    SOLVER.WEIGHT_DECAY "0.0001" \
#    MODEL.MODEL_ROOT "${model_root}" \
#    DATA.DATAPATH "${data_path}" \
#    OUTPUT_DIR "${output_dir}/seed${seed}"
#done

python visual_prompt_tuning/tune_vtab.py \
  --config-file visual_prompt_tuning/configs/prompt/cub.yaml \
  MODEL.TYPE "vit" \
  DATA.BATCH_SIZE "64" \
  MODEL.PROMPT.NUM_TOKENS "50" \
  MODEL.PROMPT.DEEP "True" \
  MODEL.PROMPT.DROPOUT "0.1" \
  DATA.FEATURE "sup_vitb16_imagenet21k" \
  DATA.NAME "vtab-${dataset}" \
  DATA.NUMBER_CLASSES "${num_classes}" \
  MODEL.MODEL_ROOT "${model_root}" \
  DATA.DATAPATH "${data_path}" \
  OUTPUT_DIR = "${output_dir}"