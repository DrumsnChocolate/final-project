#!/bin/bash
#SBATCH -J moving_data
#SBATCH -c 1
#SBATCH --mail-type=END,FAIL

# this script is to move data from personal storage to local storage of the node
# this could speed up data loading times significantly, and with it, experiment duration.

mkdir /local/${SLURM_JOBID}
## copy data to node
cd ${SLURM_SUBMIT_DIR}/implementation/mmsegmentation
cp -rL --parents data/cbis /local/${SLURM_JOBID}

## execute the pyton we want


## clean up

rm -rf /local/${SLURM_JOBID}
