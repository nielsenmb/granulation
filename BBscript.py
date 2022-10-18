#!/usr/bin/env python
# coding: utf-8

import os, sys
import pandas as pd
from asy_bkg_fitting import spectrum_fit 
from matplotlib.pyplot import *
rcParams['font.size'] = 18
 

download_dir = '/rds/projects/b/ballwh-tess-yield/data'

workDir = '/rds/projects/n/nielsemb-plato-peakbagging/granulation/'
 
prior_data_fname = os.path.join(*[workDir, 'bkgfit_output_nopca.csv']) 

prior_data = pd.read_csv(prior_data_fname)
 
figM, axM = subplots(1, 2, figsize=(16,9))
# fig3, ax3 = subplots(3, 3, figsize=(9,9))
# fig4, ax4 = subplots(4, 4, figsize=(12,12))
# figA, axA = subplots(13, 13, figsize=(32,32))

cornerN = 5000

i = int(sys.argv[1]) - 1

pcadim = int(sys.argv[2])

if sys.argv[3] == 'True':
    clear = True
else:
    clear = False

ID = prior_data.loc[i, 'ID']

print(f'{ID}')

# if pcadim > 0:
#    print(f'Running with {pcadim} pca dimensions')


# Establish output dir
outputDir = os.path.join(*[workDir, 'results', ID])

if not os.path.exists(outputDir):
   os.makedirs(outputDir)


# Check for pre-existing runs. Clear if needed.
if pcadim > 0:
    ext = f'_pca{pcadim}'
else:
    ext = '_nopca'

# fnames = {'full_sample': os.path.join(*[outputDir, ID+f'_full_sample{ext}.npz'])}

# if os.path.exists(fnames['full_sample']):
#     if clear:
#         os.remove(fnames['full_sample'])
#         print('Removing %s' % (os.path.basename(fnames['full_sample'])))
#     else:
#         print(f'{ID} already done, ending')
#         sys.exit()


# Start setup
_numax = prior_data.loc[i, 'numax'] # tgt numax
_teff = prior_data.loc[i, 'teff'] # tgt numax
_bp_rp = prior_data.loc[i, 'bp_rp'] # tgt numax
_dnu = prior_data.loc[i, 'dnu'] # tgt numax


obs = {'numax': [10**_numax, 0.01*10**_numax], 
       'teff': [10**_teff, 100],
       'bp_rp': [_bp_rp, 0.1],
       'dnu': [10**_dnu, 0.01*10**_dnu]} 

sfit = spectrum_fit(ID, obs, download_dir, pcadim=pcadim, N=200, fname=prior_data_fname)

sfit.plotModel(figM, axM, obs=obs, outputDir=outputDir);
axM.clear()

print('Running the sampler')
dynSampler, dynSamples = sfit.runDynesty(progress=False)

sfit.storeResults(outputDir)

sfit.plotModel(figM, axM, dynSamples, obs=obs, outputDir=outputDir)
axM.clear()

print('Done')

