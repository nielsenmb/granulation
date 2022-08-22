from granulation_fitting import scalingRelations
import pandas as pd
import numpy as np
import os, sys, dill
 
workDir = '/home/nielsemb/work/repos/granulation'

prior_data = pd.read_csv(os.path.join(*[workDir, 'prior_data.csv']))

scr = scalingRelations()

updated_data = prior_data.copy()

new_keys = ['H1_power', 'H1_nu', 'H1_exp',
            'H2_power', 'H2_nu', 'H2_exp',
            'H3_power', 'H3_nu', 'H3_exp',
            'bkg_numax', 'bkg_envWidth', 'bkg_envHeight',
            'shot']

for key in new_keys:
    updated_data[key] = np.nan
    
    updated_data[key+'_err'] = np.nan

updated_data['completed'] = 0

start = int(sys.argv[1])
stop = int(sys.argv[2])

percentiles= np.array([0.159, 0.5, 0.841])

for i in prior_data.index[start: stop]:
    
    ID = prior_data.loc[i, 'ID']
     
    try:
        path = os.path.join(*[workDir, 'results', ID, f'{ID}.gfit'])

        with open(path, "rb") as inputfile:
            gfit = dill.load(inputfile)

    except:
        print(i, f'{ID} samples not found.')
        continue
    
    nsamples = gfit._samples.shape[0]
    
    ndim = len(gfit.labels)
    
    full_samples = np.zeros((nsamples, ndim))
    
    for k in range(nsamples):
        full_samples[k, :] = gfit.unpackParams(gfit._samples[k, :])
        
    full_samples = np.log10(full_samples)
   
    for j in range(ndim):
        smp = full_samples[:, j]
        
        percs = np.percentile(smp, percentiles)

        updated_data.at[i, new_keys[j]] = percs[1]

        updated_data.at[i, new_keys[j]+'_err'] = np.mean(np.diff(percs))

    updated_data.at[i, 'completed'] = 1 

updated_data.to_csv(os.path.join(*[workDir, 'bkgfit_output.csv']), index=False)