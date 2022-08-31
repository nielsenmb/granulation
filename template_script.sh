#!/bin/bash
#SBATCH --ntasks 4
#SBATCH --time TIME
#SBATCH --qos bbdefault
#SBATCH --mail-type NONE
#SBATCH --job-name=PID
#SBATCH --account=nielsemb-plato-peakbagging
#SBATCH --array START-END
#SBATCH --output /rds/projects/n/nielsemb-plato-peakbagging/granulation/slurm_output/slurm-%A_%a.out

set -e

module purge; module load bluebear 
module load bear-apps/2019b
module load GCC/8.3.0
module load Theano/1.0.4-foss-2019b-Python-3.7.4
module load Python/3.7.4-GCCcore-8.3.0

source /rds/homes/n/nielsemb/.virtualenvs/peakbogging/bin/activate

echo "${SLURM_JOB_ID}: Job ${SLURM_ARRAY_TASK_ID} of ${SLURM_ARRAY_TASK_MAX} in the array"
python -u /rds/projects/n/nielsemb-plato-peakbagging/granulation/BBscript.py ${SLURM_ARRAY_TASK_ID} NPCA

 

