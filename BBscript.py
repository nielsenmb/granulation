#!/usr/bin/env python
# coding: utf-8

import os, sys, traceback
import pandas as pd
import numpy as np
from granulation_fitting import granulation_fit 
from matplotlib.pyplot import *
rcParams['font.size'] = 18

# Set index range of interest

download_dir = '/rds/projects/b/ballwh-tess-yield/data'

workDir = '/rds/projects/n/nielsemb-plato-peakbagging/granulation/'

prior_data = pd.read_csv(os.path.join(*[workDir, 'outlier_run.csv']))
 
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

# if os.path.exists(os.path.join(*[outputDir, ID+'_samples.npz'])):
#    print(f'{ID} already done, ending')
#    sys.exit()

_numax = prior_data.loc[i, 'numax']
numax = (10**_numax, 0.2*10**_numax)

print('Initializing fit class')
gfit = granulation_fit(ID, numax, download_dir)

gfit.plotModel(figM, axM, outputDir=outputDir);
axM.clear()

print('Running the sampler')
sampler, samples = gfit.runDynesty()

gfit.storeResults(samples, outputDir)

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

print('Done')

