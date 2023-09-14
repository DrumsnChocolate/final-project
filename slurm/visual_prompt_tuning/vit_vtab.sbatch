#!/bin/bash
#SBATCH -J vpt-vit-vtab
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL

# copy visual prompt tuning folder to node (beware of running experiments that involve symlinked data, that might not go well)
cd /local
mkdir ${SLURM_JOBID}
cp -r ${SLURM_SUBMIT_DIR}/implementation/visual_prompt_tuning .

# setup environment with conda
env_setup.sh

# todo: replace below path with proper path on node
visual_prompt_tuning/experiments/vit_vtab.sh

env_teardown.sh

# copy output back to submit directory
cp -r visual_prompt_tuning/output_dir ${SLURM_SUBMIT_DIR}/implementation/visual_prompt_tuning

# remove newly created folder
rm -r *
cd ..
rmdir ${SLURM_JOBID}

# Done.



