#!/bin/bash
#SBATCH -J test-setr-untested
#SBATCH -c 8
#SBATCH --gres=gpu:ampere:1
#SBATCH --mail-type=END,FAIL

cd implementation/mmsegmentation
source ~/miniconda3/etc/profile.d/conda.sh
conda activate openmmlab

# list all the setr runs so far, by matching work_dirs/*/202*/last_checkpoint.pth
# echo
# echo "List of all setr runs so far:"

for i in $(find work_dirs/*/202* -name last_checkpoint); do

  # obtain directory name from path
  dir=$(dirname $i)

  # if not test folder exists for this run, test it
    if [ ! -d "$dir/test" ]; then
      python tools/test.py $dir
    fi
done