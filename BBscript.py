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

ID = prior_data.loc[i, 'ID']

print(f'{ID}')

outputDir = os.path.join(*[workDir, 'results', ID])

if not os.path.exists(outputDir):
   os.makedirs(outputDir)

# if os.path.exists(os.path.join(*[outputDir, ID+'_full_sample.npz'])):
#    print(f'{ID} already done, ending')
#    sys.exit()

_numax = prior_data.loc[i, 'numax']
numax = (10**_numax, 0.1*10**_numax)

print('Initializing fit class')
gfit = granulation_fit(ID, numax, download_dir, pcadim=9, weights=wfunc, weight_args={'n':2}, N=200)

gfit.plotModel(figM, axM, outputDir=outputDir);
axM.clear()

print('Running the sampler')
sampler, samples = gfit.runDynesty()

gfit.storeResults(outputDir)

gfit.plotModel(figM, axM, samples, outputDir=outputDir)
axM.clear()

print('Done')

