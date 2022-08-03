#!/usr/bin/env python3
import pandas as pd
import numpy as np
import glob
import os

scriptfiles = glob.glob('scripts/script_*.sh')

for f in scriptfiles:
    try:
        os.remove(f)
    except OSError as e:
        print("Error: %s : %s" % (f, e.strerror))

batches = 5
njobs = len(pd.read_csv('prior_data.csv'))

if batches > njobs:
    batches = njobs


with open('template_script.sh') as fin:
    template = fin.read()

njobs_per_batch = np.floor(njobs / batches)
if njobs_per_batch < 1:
    njobs_per_batch = 1

print(f'Njobs : {njobs}, Nbatches : {batches}')

start = np.arange(batches + 1) * njobs_per_batch

end = start + njobs_per_batch

#hrs_per_tgt = 0.3

for idx, st in enumerate(start):
    sr = template.replace('START', str(int(st+1))).replace('END', str(int(st + 1 + njobs_per_batch)))

    sr = sr.replace('TIME', '1:0:0') # % (int(njobs_per_batch*hrs_per_tgt)))

    sr = sr.replace('IDX', str(int(st)))

    sr = sr.replace('PID', 'P'+str(int(st)))

    with open(f'scripts/script_{idx}.sh', 'w') as fout:
        fout.write(sr)
