#!/bin/bash
#SBATCH -c 2
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40

cd implementation/segment_anything
source ~/miniconda3/etc/profile.d/conda.sh
conda activate segment_anything

for i in $(find outputs/* -name checkpoint.pth); do
  dir=$(dirname $i)
  if [ ! -d "$dir/test" ]; then
    python finetune/test.py $dir
  fi
done

