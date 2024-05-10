#!/bin/bash
#SBATCH -J zgt_full-tuned-cbis
#SBATCH -c 2
#SBATCH --mail-type=START,END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40
#SBATCH --time=0-01:00:00


# evaluate the cbis full tuned model on the zgt dataset
cd implementation/segment_anything
source ~/miniconda3/etc/profile.d/conda.sh
conda activate segment_anything

timestamp="20240416_104218"  # yyyymmdd_hhmmss

python finetune/test.py outputs/${timestamp} --sub-dir 'test_zgt' --cfg-options data.root="data/zgt/zgt-linked" data.test="{annotation_dir: 'annotations_binary', batch_size: 1, image_dir: 'images'}"

conda deactivate


