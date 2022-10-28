import os, sys
import pandas as pd
from asy_bkg_fitting import spectrum_fit 
from matplotlib.pyplot import *
import numpy as np
rcParams['font.size'] = 18

download_dir = '/rds/projects/b/ballwh-tess-yield/data'

workDir = '/rds/projects/n/nielsemb-plato-peakbagging/granulation/'
 
prior_data_fname = os.path.join(*[workDir, 'bkgfit_output_nopca.csv']) 

prior_data = pd.read_csv(prior_data_fname)

pcadim = int(sys.argv[1])

test_numaxs = np.linspace(min(prior_data['numax']), 
                          max(prior_data['numax']), 50)

idxs = np.array([np.argmin(abs(prior_data['numax'].values - nu)) for nu in test_numaxs])

ext = f'_pca{pcadim}'

for i in idxs:
    ID = prior_data.loc[i, 'ID']

    # Establish output dir
    outputDir = os.path.join(*[workDir, 'results', ID])

    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
 
    _numax = prior_data.loc[i, 'numax']
    _teff = prior_data.loc[i, 'teff']
    _bp_rp = prior_data.loc[i, 'bp_rp'] 


    obs = {'numax': [10**_numax, 0.01*10**_numax], 
           'teff': [10**_teff, 100],
           'bp_rp': [_bp_rp, 0.1]} 

    ext = f'pca{pcadim}'
    
    fname = os.path.join(*[outputDir, ID + f'_{ext}.sfit'])

    if os.path.exists(fname):
        continue

    sfit = spectrum_fit(ID, obs, download_dir, pcadim=pcadim, N=200, fname=prior_data_fname)
        
    dynSampler, dynSamples = sfit.runDynesty(progress=False)

    sfit.storeResults(outputDir)