#!/usr/bin/env bash
./implementation/visual_prompt_tuning/env_setup.sh

mkdir -p implementation/visual_prompt_tuning/data_path  # folder to put any datasets for vpt, symlinks are also fine
mkdir -p implementation/visual_prompt_tuning/model_root  # folder to put model roots for vpt. explained in visual_prompt_tuning/README.md
mkdir -p implementation/visual_prompt_tuning/output_dir  # output directory for vpt


./implementation/mmsegmentation/env_setup.sh
# todo: rest of mmseg setup