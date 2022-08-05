#!/bin/bash
#SBATCH --ntasks 1
#SBATCH --time 4:0:0
#SBATCH --qos bbdefault
#SBATCH --mail-type NONE
#SBATCH --job-name=PID
#SBATCH --account=nielsemb-plato-peakbagging

set -e

module purge; module load bluebear 
module load bear-apps/2019b
module load GCC/8.3.0
module load Theano/1.0.4-foss-2019b-Python-3.7.4
module load Python/3.7.4-GCCcore-8.3.0

source /rds/homes/n/nielsemb/.virtualenvs/peakbogging/bin/activate

python -u /rds/projects/n/nielsemb-plato-peakbagging/granulation/download.py 10000 14000