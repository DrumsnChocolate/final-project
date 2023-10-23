#!/bin/bash
#SBATCH -c 1
#SBATCH --mail-type=END,FAIL

## copy visual prompt tuning folder to node (beware of running experiments that involve symlinked data, that might not go well)
#cd /local
#mkdir ${SLURM_JOBID}
#cd ${SLURM_JOBID}
#cp -r ${SLURM_SUBMIT_DIR}/implementation/visual_prompt_tuning .

cd implementation

# properly load conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate prompt

# launch experiments
visual_prompt_tuning/experiments/vit_mammo.sh $1 $2 $3 $4 $5

conda deactivate

## copy output back to submit directory
#cp -r visual_prompt_tuning/output_dir ${SLURM_SUBMIT_DIR}/implementation/visual_prompt_tuning
#
## remove newly created folder
#rm -r *
#cd ..
#rmdir ${SLURM_JOBID}

# Done.



