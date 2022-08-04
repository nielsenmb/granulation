#!/usr/bin/env python
# coding: utf-8

import os, sys
import pandas as pd
import numpy as np
from granulation_fitting import granulation_fit 
from matplotlib.pyplot import *
rcParams['font.size'] = 18

# Set index range of interest

download_dir = '/rds/projects/b/ballwh-tess-yield/data'

workDir = '/rds/projects/n/nielsemb-plato-peakbagging/background_fit/'

prior_data = pd.read_csv(os.path.join(*[workDir, 'prior_data.csv']))

updated_data = prior_data.copy()

new_keys = ['H1_power', 'H1_nu', 'H1_exp',
            'H2_power', 'H2_nu', 'H2_exp',
            'H3_power', 'H3_nu', 'H3_exp',
            'bkg_numax', 'bkg_envWidth', 'bkg_envHeight',
            'shot']

for key in new_keys:
    updated_data[key] = np.nan
    updated_data[key+'_err'] = np.nan

figM, axM = subplots(figsize=(16,9))
fig3, ax3 = subplots(3, 3, figsize=(9,9))
fig4, ax4 = subplots(4, 4, figsize=(12,12))
figA, axA = subplots(13, 13, figsize=(32,32))

cornerN = 5000

#indcs = np.array(sys.argv[1:], dtype=int)
#for i in prior_data.index[indcs[0]:indcs[1]]:

i = int(sys.argv[1]) - 1

ID = prior_data.loc[i, 'ID']

print(f'{ID}')

outputDir = os.path.join(*[workDir, 'results', ID])

# if os.path.exists(os.path.join(*[outputDir, ID+'_samples_1.npz'])):
#     print(f'{ID} already done, ending')
#     continue

if not os.path.exists(outputDir):
    os.makedirs(outputDir)

#try:
_numax = prior_data.loc[i, 'numax']
numax = (10**_numax, 0.2*10**_numax)

gfit = granulation_fit(ID, numax, download_dir)

gfit.plotModel(figM, axM, outputDir=outputDir);
axM.clear()

sampler, samples = gfit.runDynesty()

gfit.storeResults(i, updated_data, new_keys, samples, outputDir)

gfit.plotModel(figM, axM, samples, outputDir=outputDir)
axM.clear()

gfit.makeCorner(fig3, samples, cornerN, labels=['hsig1', 'dhnu1', 'hexp1'], outputDir=outputDir);
for ax in ax3.flatten():
   ax.clear()

gfit.makeCorner(fig3, samples, cornerN, labels=['hsig2', 'dhnu2', 'hexp2'], outputDir=outputDir);
for ax in ax3.flatten():
   ax.clear()

gfit.makeCorner(fig3, samples, cornerN, labels=['hsig3', 'hnu3', 'hexp3'], outputDir=outputDir);
for ax in ax3.flatten():
   ax.clear()

gfit.makeCorner(fig4, samples, cornerN, labels=['numax', 'dwidth', 'height', 'white'], outputDir=outputDir);
for ax in ax4.flatten():
   ax.clear()

gfit.makeCorner(figA, samples, cornerN, labels=gfit.labels, outputDir=outputDir);
for ax in axA.flatten():
   ax.clear()

# except:
#     try:
#         exc_info = sys.exc_info()

#     finally:
#         # Display the *original* exception
#         traceback.print_exception(*exc_info)
#         del exc_info
#     print(f'{ID} failed, going to next tgt')
#     continue

print('Done')

