import pandas as pd
import numpy as np
import os, sys
 
#download_dir = '/home/nielsemb/work/mounts/Bluebear_data/data'
download_dir = '/rds/projects/b/ballwh-tess-yield/data'

#workDir = '/home/nielsemb/work/mounts/Bluebear_projects/granulation'
workDir = '/rds/projects/n/nielsemb-plato-peakbagging/granulation/'

prior_data = pd.read_csv(os.path.join(*[workDir, 'prior_data.csv']))
 
new_keys = ['bkg_numax', 'bkg_envHeight', 'bkg_envWidth', 
            'H1_power', 'H1_nu', 'H1_exp',
            'H2_power', 'H2_nu', 'H2_exp',
            'H3_power', 'H3_nu', 'H3_exp',
            'shot']

for key in new_keys:
    prior_data[key] = np.nan
    
    prior_data[key+'_err'] = np.nan

prior_data['completed'] = 0

start = int(sys.argv[1])
stop = int(sys.argv[2])

for i in prior_data.index[start: stop]:
    
    ID = prior_data.loc[i, 'ID']
    print(ID)

    try:
        full_samples = np.load(os.path.join(*[workDir, 'results', ID, f'{ID}_full_samples.npz']))['samples']
    except:
        print(i, f'{ID} samples not found.')
        continue
    
    full_samples = np.log10(full_samples)
    
    # numax, height, width, hsig1, hnu1, exp1, hsig2, hnu2, exp2, hsig3, hnu3, exp3, w
    percentiles= np.array([0.159, 0.5, 0.841]) * 100

    for j in range(full_samples.shape[1]):

        smp = full_samples[:, j]
        
        percs = np.percentile(smp, percentiles)

        prior_data.at[i, new_keys[j]] = percs[1]

        prior_data.at[i, new_keys[j]+'_err'] = np.mean(np.diff(percs))

    prior_data.at[i, 'completed'] = 1 

prior_data.to_csv(os.path.join(*[workDir, 'bkgfit_output_w_pca9.csv']), index=False)