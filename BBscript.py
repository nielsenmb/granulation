#!/usr/bin/env python
# coding: utf-8

import os, sys
import pandas as pd
from granulation_fitting import granulation_fit 
from matplotlib.pyplot import *
rcParams['font.size'] = 18

def wfunc(self, n=1):
     
    ppf, pdf = self.getQuantileFuncs(self.data_F[:, :1])

    w = 1/pdf[0](self.data_F[:, 0])**n
       
    return w

download_dir = '/rds/projects/b/ballwh-tess-yield/data'

workDir = '/rds/projects/n/nielsemb-plato-peakbagging/granulation/'

prior_data = pd.read_csv(os.path.join(*[workDir, 'prior_data.csv']))
 
figM, axM = subplots(figsize=(16,9))
fig3, ax3 = subplots(3, 3, figsize=(9,9))
fig4, ax4 = subplots(4, 4, figsize=(12,12))
figA, axA = subplots(13, 13, figsize=(32,32))

cornerN = 5000

i = int(sys.argv[1]) - 1

pcadim = int(sys.argv[2])

if sys.argv[3] == 'True':
    clear = True
else:
    clear = False

ID = prior_data.loc[i, 'ID']

print(f'{ID}')

if pcadim > 0:
   print(f'Running with {pcadim} pca dimensions')


# Establish output dir
outputDir = os.path.join(*[workDir, 'results', ID])

if not os.path.exists(outputDir):
   os.makedirs(outputDir)


# Check for pre-existing runs. Clear if needed.
if pcadim > 0:
    ext = f'_pca{pcadim}'
else:
    ext = '_nopca'

    

fnames = {'full_sample': os.path.join(*[outputDir, ID+f'_full_sample{ext}.npz'])}

if os.path.exists(fnames['full_sample']):
    if clear:
        os.remove(fnames['full_sample'])
        print('Removing %s' % (os.path.basename(fnames['full_sample'])))
    else:
        print(f'{ID} already done, ending')
        sys.exit()


# Start setup
_numax = prior_data.loc[i, 'numax']
numax = (10**_numax, 0.1*10**_numax)

gfit = granulation_fit(ID, numax, download_dir, pcadim=pcadim, weights=wfunc, weight_args={'n':2}, N=200)

gfit.plotModel(figM, axM, outputDir=outputDir);
axM.clear()

print('Running the sampler')
sampler, samples = gfit.runDynesty()

gfit.storeResults(outputDir)

gfit.plotModel(figM, axM, samples, outputDir=outputDir)
axM.clear()

print('Done')

