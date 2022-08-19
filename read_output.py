from granulation_fitting import scalingRelations
import pandas as pd
import numpy as np
import os, sys
import jax.numpy as jnp

download_dir = '/rds/projects/b/ballwh-tess-yield/data'

workDir = '/rds/projects/n/nielsemb-plato-peakbagging/granulation/'

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

for i in prior_data.index[int(sys.argv[1]): int(sys.argv[2])]:
    
    ID = prior_data.loc[i, 'ID']
    
    try:
        samples = np.load(os.path.join(*[workDir, 'results', ID, f'{ID}_samples.npz']))['samples']
    except:
        print(i, f'{ID} samples not found.')
        continue

    hsig1, dhnu1, exp1, hsig2, dhnu2, exp2, hsig3, hnu3, exp3, numax, dwidth, height, white = samples.T

    hnu1 = scr.nuHarveyEnv(numax) * dhnu1

    hnu2 = scr.nuHarveyGran(numax) * dhnu2

<<<<<<< HEAD
    width = dwidth * scr.envWidth(numax) * (1/ (2 * jnp.sqrt(2 * jnp.log(2))) / 2)
=======
    width = dwidth * scr.envWidth(numax) * (1 / (2 * jnp.sqrt(2 * jnp.log(2))) / 2)
>>>>>>> 46dbd3143397f22df74f8463a4d0bdd5eb23a00c

    percentiles= np.array([0.159, 0.5, 0.841])

    pars = [hsig1, np.log10(hnu1), np.log10(exp1),
            hsig2, np.log10(hnu2), np.log10(exp2),
            hsig3, np.log10(hnu3), np.log10(exp3),
            np.log10(numax), np.log10(width), height,
            white]

    for j, smp in enumerate(pars):

        percs = np.percentile(smp, percentiles)

        updated_data.at[i, new_keys[j]] = percs[1]

        updated_data.at[i, new_keys[j]+'_err'] = np.mean(np.diff(percs))

    updated_data.at[i, 'completed'] = 1 

updated_data.to_csv(os.path.join(*[workDir, 'bkgfit_output.csv']), index=False)