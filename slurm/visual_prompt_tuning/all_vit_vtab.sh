#!/bin/bash
command="visual_prompt_tuning/experiments/vit_vtab.sh"
# todo: include last dataset after manual download
#for d in "caltech101 102" "cifar(num_classes=100) 100" "dtd 47" "oxford_flowers102 102" "oxford_iiit_pet 37" "patch_camelyon 2" "sun397 397" "svhn 10" "resisc45 45" "eurosat 10" "dmlab 6" "kitti(task=\"closest_vehicle_distance\") 4" "smallnorb(predicted_attribute=\"label_azimuth\") 18" "smallnorb(predicted_attribute=\"label_elevation\") 9" "dsprites(predicted_attribute=\"label_x_position\",num_classes=16) 16" "dsprites(predicted_attribute=\"label_orientation\",num_classes=16) 16" "clevr(task=\"closest_object_distance\") 6" "clevr(task=\"count_all\") 8" "diabetic_retinopathy(config=\"btgraham-300\") 5"; do
# todo: first test for one dataset, and then proceed to the rest
#for d in "cifar(num_classes=100) 100" "dtd 47" "oxford_flowers102 102" "oxford_iiit_pet 37" "patch_camelyon 2" "sun397 397" "svhn 10" "dmlab 6" "kitti(task=\"closest_vehicle_distance\") 4" "smallnorb(predicted_attribute=\"label_elevation\") 9" "dsprites(predicted_attribute=\"label_x_position\",num_classes=16) 16" "dsprites(predicted_attribute=\"label_orientation\",num_classes=16) 16" "clevr(task=\"closest_object_distance\") 6" "clevr(task=\"count_all\") 8" "diabetic_retinopathy(config=\"btgraham-300\") 5"; do
#for d in "dsprites(predicted_attribute=\"label_x_position\",num_classes=16) 16"; do
for d in "caltech101 102" "cifar(num_classes=100) 100" "dtd 47" "oxford_flowers102 102" "oxford_iiit_pet 37" "patch_camelyon 2" "sun397 397" "svhn 10" "resisc45 45" "eurosat 10" "dmlab 6" "kitti(task=\"closest_vehicle_distance\") 4" "smallnorb(predicted_attribute=\"label_azimuth\") 18" "smallnorb(predicted_attribute=\"label_elevation\") 9" "dsprites(predicted_attribute=\"label_x_position\",num_classes=16) 16" "dsprites(predicted_attribute=\"label_orientation\",num_classes=16) 16" "clevr(task=\"closest_object_distance\") 6" "clevr(task=\"count_all\") 8" "diabetic_retinopathy(config=\"btgraham-300\") 5"; do  set -- $d
  dataset=$1
  num_classes=$2
  sbatch -J "vpt-vit-vtab-${dataset}" slurm/visual_prompt_tuning/vpt_experiment.sbatch $command $dataset $num_classes
done
