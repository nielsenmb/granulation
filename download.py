import lightkurveCacheAccess as lka
import pandas as pd
import os, sys, traceback

download_dir = '/rds/projects/b/ballwh-tess-yield/data'

workDir = '/rds/projects/n/nielsemb-plato-peakbagging/granulation/'

prior_data = pd.read_csv(os.path.join(*[workDir, 'prior_data.csv']))


for i in prior_data.index[int(sys.argv[1]): int(sys.argv[2])]:

    ID = prior_data.loc[i, 'ID']
    print(i, ID)
    _numax = prior_data.loc[i, 'numax']
    numaxGuess = (10**_numax, 0.2*10**_numax)
    
    lk_kwargs = {}

    if 'KIC' in ID:
        lk_kwargs['author'] = 'Kepler'
        lk_kwargs['mission'] = 'Kepler'

        if numaxGuess[0] > 1/(2*1800)*1e6:
            lk_kwargs['exptime'] = 60
        else:
            lk_kwargs['exptime'] = 1800

    if 'TIC' in ID:
        lk_kwargs['author'] = 'SPOC'
        lk_kwargs['mission'] = 'TESS'
        lk_kwargs['exptime'] = 120
    try:
        lka.search_lightcurve(ID, download_dir, lk_kwargs, use_cached=True, cache_expire=10*365)
    except:
        try:
         exc_info = sys.exc_info()

        finally:
            # Display the *original* exception
            traceback.print_exception(*exc_info)
            del exc_info
        print(f'{ID} failed to download, going to next tgt')
        continue
