#!/bin/bash
#SBATCH -c 1
#SBATCH --gres=gpu:ampere:1
#SBATCH --mail-type=END,FAIL

command=$1
dataset=$2
num_classes=$3

cd implementation

# properly load conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate prompt

# launch experiments
$command $dataset $num_classes "prompt"

# deactivate conda
conda deactivate


