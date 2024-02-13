#!/bin/bash
#SBATCH -J moving_data
#SBATCH -c 1
#SBATCH --mail-type=END,FAIL

# this script is to move data from personal storage to local storage of the node
# this could speed up data loading times significantly, and with it, experiment duration.

# let's first inspect where the data is to be stored
echo $SLURM_TMPDIR
