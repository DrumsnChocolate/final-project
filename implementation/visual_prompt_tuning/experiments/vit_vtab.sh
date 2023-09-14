model_root=visual_prompt_tuning/model_root
data_path=visual_prompt_tuning/data_path
output_dir=visual_prompt_tuning/output_dir

# todo: include last dataset after manual download
#for d in "caltech101 102" "cifar(num_classes=100) 100" "dtd 47" "oxford_flowers102 102" "oxford_iiit_pet 37" "patch_camelyon 2" "sun397 397" "svhn 10" "resisc45 45" "eurosat 10" "dmlab 6" "kitti(task=\"closest_vehicle_distance\") 4" "smallnorb(predicted_attribute=\"label_azimuth\") 18" "smallnorb(predicted_attribute=\"label_elevation\") 9" "dsprites(predicted_attribute=\"label_x_position\",num_classes=16) 16" "dsprites(predicted_attribute=\"label_orientation\",num_classes=16) 16" "clevr(task=\"closest_object_distance\") 6" "clevr(task=\"count_all\") 8" "diabetic_retinopathy(config=\"btgraham-300\") 5"; do
# todo: first test for one dataset, and then proceed to the rest
#for d in "caltech101 102" "cifar(num_classes=100) 100" "dtd 47" "oxford_flowers102 102" "oxford_iiit_pet 37" "patch_camelyon 2" "sun397 397" "svhn 10" "resisc45 45" "eurosat 10" "dmlab 6" "kitti(task=\"closest_vehicle_distance\") 4" "smallnorb(predicted_attribute=\"label_azimuth\") 18" "smallnorb(predicted_attribute=\"label_elevation\") 9" "dsprites(predicted_attribute=\"label_x_position\",num_classes=16) 16" "dsprites(predicted_attribute=\"label_orientation\",num_classes=16) 16" "clevr(task=\"closest_object_distance\") 6" "clevr(task=\"count_all\") 8"; do
for d in  "caltech101 102"; do
  set -- $d
  dataset=$1
  num_classes=$2

  # parameters are specified based on the visual prompt tuning paper, and the hyper parameter tuning that was done there
  for seed in "42" "44" "82" "100" "800"; do
    python visual_prompt_tuning/train.py \
      --config-file visual_prompt_tuning/configs/prompt/cub.yaml \
      MODEL.TYPE "vit" \
      DATA.BATCH_SIZE "64" \
      MODEL.PROMPT.NUM_TOKENS "100" \
      MODEL.PROMPT.DEEP "True" \
      MODEL.PROMPT.DROPOUT "0.1" \
      DATA.FEATURE "sup_vitb16_imagenet21k" \
      DATA.NAME "vtab-${dataset}" \
      DATA.NUMBER_CLASSES "${num_classes}" \
      SOLVER.BASE_LR "5.0" \
      SOLVER.WEIGHT_DECAY "0.0001" \
      MODEL.MODEL_ROOT "${model_root}" \
      DATA.DATAPATH "${data_path}" \
      OUTPUT_DIR "${output_dir}/seed${seed}"
  done
done