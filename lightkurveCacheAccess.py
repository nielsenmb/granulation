import os, pickle, re
import lightkurve as lk
from datetime import datetime

def format_name(ID):
    """ Format input ID
    
    Users tend to be inconsistent in naming targets, which is an issue for 
    looking stuff up on, e.g., Simbad. This function formats the ID so that 
    Simbad doesn't throw a fit.
    
    If the name doesn't look like anything in the variant list it will only be 
    changed to a lower-case string.
    
    Parameters
    ----------
    ID : str
        Name to be formatted.
    
    Returns
    -------
    ID : str
        Formatted name
        
    """

    ID = str(ID)
    ID = ID.lower()
    
    # Add naming exceptions here
    variants = {'KIC': ['kic', 'kplr', 'KIC'],
                'Gaia DR2': ['gaia dr2', 'gdr2', 'dr2', 'Gaia DR2'],
                'Gaia DR1': ['gaia dr1', 'gdr1', 'dr1', 'Gaia DR1'], 
                'EPIC': ['epic', 'ktwo', 'EPIC'],
                'TIC': ['tic', 'tess', 'TIC']
               }
    
    fname = None
    for key in variants:   
        for x in variants[key]:
            if x in ID:
                fname = ID.replace(x,'')
                fname = re.sub(r"\s+", "", fname, flags=re.UNICODE)
                fname = key+' '+str(int(fname))
                return fname 
    return ID

def load_fits(files, mission):
    """ Read fitsfiles into a Lightkurve object
    
    Parameters
    ----------
    files : list
        List of pathnames to fits files
    mission : str
        Which mission to download the data from.
    
    Returns
    -------
    lc : lightkurve.lightcurve.KeplerLightCurve object
        Lightkurve light curve object containing the concatenated set of 
        quarters.
        
    """
    if mission in ['Kepler', 'K2']:
        lcs = [lk.lightcurvefile.KeplerLightCurveFile(file) for file in files]
        lcCol = lk.LightCurveCollection(lcs)
        #lc = lccol.PDCSAP_FLUX.stitch()
    elif mission in ['TESS']:
        lcs = [lk.lightcurvefile.TessLightCurveFile(file) for file in files]
        lcCol = lk.LightCurveCollection(lcs)
        #lc = lccol.PDCSAP_FLUX.stitch()
    return lcCol

def set_mission(ID, lkwargs):
    """ Set mission keyword in lkwargs.
    
    If no mission is selected will attempt to figure it out based on any
    prefixes in the ID string, and add this to the LightKurve keywords 
    arguments dictionary.
    
    Parameters
    ----------
    ID : str
        ID string of the target
    lkwargs : dict
        Dictionary to be passed to LightKurve
        
    """

    if lkwargs['mission'] is None:
        if ('kic' in ID.lower()):
            lkwargs['mission'] = 'Kepler'
        elif ('epic' in ID.lower()) :
            lkwargs['mission'] = 'K2'
        elif ('tic' in ID.lower()):
            lkwargs['mission'] = 'TESS'
        else:
            lkwargs['mission'] = ('Kepler', 'K2', 'TESS')
            
def search_and_dump(ID, lkwargs, search_cache):
    """ Get lightkurve search result online.
    
    Uses the lightkurve search_lightcurve to find the list of available data 
    for a target ID. 
    
    Stores the result in the ~/.lightkurve-cache/searchResult directory as a 
    dictionary with the search result object and a timestamp.
    
    Parameters
    ----------
    ID : str
        ID string of the target
    lkwargs : dict
        Dictionary to be passed to LightKurve
    search_cache : str
        Directory to store the search results in. 
        
    Returns
    -------
    resultDict : dict
        Dictionary with the search result object and timestamp.    
    """
    
    current_date = datetime.now().isoformat()
    store_date = current_date[:current_date.index('T')].replace('-','')

    search = lk.search_lightcurve(ID, exptime=lkwargs['exptime'], 
                                  mission=lkwargs['mission'])
    resultDict = {'result': search,
                  'timestamp': store_date}
    
    fname = os.path.join(*[search_cache, f"{ID}_{lkwargs['exptime']}.lksearchresult"])
    
    pickle.dump(resultDict, open(fname, "wb"))
    
    return resultDict   

def getMASTidentifier(ID, lkwargs):
    """ return KIC/TIC/EPIC for given ID.
    
    If input ID is not a KIC/TIC/EPIC identifier then the target is looked up
    on MAST and the identifier is retried. If a mission is not specified the 
    set of observations with the most quarters/sectors etc. will be used. 
    
    Parameters
    ----------
    ID : str
        Target ID
    lkwargs : dict
        Dictionary with arguments to be passed to lightkurve. In this case
        mission and exptime.
    
    Returns
    -------
    ID : str
        The KIC/TIC/EPIC ID of the target.    
    """
    
    if not any([x in ID for x in ['KIC', 'TIC', 'EPIC']]):
        
        search = lk.search_lightcurvefile(ID, exptime=lkwargs['exptime'], mission=lkwargs['mission'])

        if len(search) == 0:
            raise ValueError(f'No results for {ID} found on MAST')

        maxFreqName = max(set(list(search.table['target_name'])), key = list(search.table['target_name']).count)
        maxFreqObsCol = max(set(list(search.table['obs_collection'])), key = list(search.table['obs_collection']).count)

        if maxFreqObsCol == 'TESS':
            prefix = 'TIC'
        else:
            prefix = ''

        temp_id = prefix + maxFreqName

        ID = format_name(temp_id).replace(' ', '')
        lkwargs['mission'] = maxFreqObsCol
    else:
        ID = ID.replace(' ', '')
    return ID

def check_sr_cache(ID, lkwargs, use_cached=True, download_dir=None, 
                   cache_expire=30):
    """ check search results cache
    
    Preferentially accesses cached search results, otherwise searches the 
    MAST archive.
    
    Parameters
    ----------
    ID : str
        Target ID (must be KIC, TIC, or ktwo prefixed)
    lkwargs : dict
        Dictionary with arguments to be passed to lightkurve. In this case
        mission and exptime.
    use_cached : bool, optional
        Whether or not to use the cached time series. Default is True.
    download_dir : str, optional.
        Directory for fits file and search results caches. Default is 
        ~/.lightkurve-cache. 
    cache_expire : int, optional.
        Expiration time for the search cache results. Files older than this 
        will be. The default is 30 days.
        
    Returns
    -------
    search : lightkurve.search.SearchResult
        Search result from MAST.  
    """
       
    # Set default lightkurve cache directory if nothing else is given
    if download_dir is None:
        download_dir = os.path.join(*[os.path.expanduser('~'), '.lightkurve-cache'])
    
    # Make the search cache dir if it doesn't exist
    cachepath = os.path.join(*[download_dir, 'searchResults', lkwargs['mission']])
    if not os.path.isdir(cachepath):
        os.makedirs(cachepath)

    filepath = os.path.join(*[cachepath, f"{ID}_{lkwargs['exptime']}.lksearchresult"])

    if os.path.exists(filepath) and use_cached:  
        
        resultDict = pickle.load(open(filepath, "rb"))
        fdate = resultDict['timestamp'] 
        ddate = datetime.now() - datetime(int(fdate[:4]), int(fdate[4:6]), int(fdate[6:]))
        
        # If file is saved more than cache_expire days ago, a new search is performed
        if ddate.days > cache_expire:   
            print(f'Last search was performed more than {cache_expire} days ago, checking for new data.')
            resultDict = search_and_dump(ID, lkwargs, cachepath)
        else:
            print('Using cached search result.')    
    else:
        print('No cached search results, searching MAST')
        resultDict = search_and_dump(ID, lkwargs, cachepath)
        
    return resultDict['result']

def check_fits_cache(search, mission, download_dir=None):
    """ Query cache directory or download fits files.
    
    Searches the Lightkurve cache directory set by download_dir for fits files
    matching the search query, and returns a list of path names of the fits
    files.
    
    If not cache either doesn't exist or doesn't contain all the files in the
    search, all the fits files will be downloaded again.
    
    Parameters
    ----------
    search : lightkurve.search.SearchResult
        Search result from MAST. 
    mission : str
        Which mission to download the data from.
    download_dir : str, optional.
        Top level of the Lightkurve cache directory. default is 
        ~/.lightkurve-cache
        
    Returns
    -------
    files_in_cache : list
        List of path names to the fits files in the cache directory
    """
        
    if download_dir is None:
        download_dir = os.path.join(*[os.path.expanduser('~'), '.lightkurve-cache'])
     
    files_in_cache = []

    for i, row in enumerate(search.table):
        fname = os.path.join(*[download_dir, 'mastDownload', mission, row['obs_id'], row['productFilename']])
        if os.path.exists(fname):
            files_in_cache.append(fname)
    
    if len(files_in_cache) != len(search):
        if len(files_in_cache) == 0:
            print('No files in cache, downloading.')
        elif len(files_in_cache) > 0:
            print('Search result did not match cached fits files, downloading.')       
        search.download_all(download_dir = download_dir)
        files_in_cache = [os.path.join(*[download_dir, 'mastDownload', mission, row['obs_id'], row['productFilename']]) for row in search.table]
    else:
        print('Loading fits files from cache.')

    return files_in_cache

def clean_lc(lc):
    """ Perform Lightkurve operations on object.

    Performes basic cleaning of a light curve, removing nans, outliers,
    median filtering etc.

    Parameters
    ----------
    lc : Lightkurve.LightCurve instance
        Lightkurve object to be cleaned

    Returns
    -------
    lc : Lightkurve.LightCurve instance
        The cleaned Lightkurve object
        
    """

    lc = lc.remove_nans().flatten(window_length=4001).remove_outliers()
    return lc

def search_lightcurve(ID, download_dir, lkwargs, use_cached, cache_expire=30):
    """ Get time series using LightKurve
    
    Performs a search for available fits files on MAST and then downloads them
    if nessary.
    
    The search results are cached with an expiration of 30 days. If a search
    result is found, the fits file cache is searched for a matching file list
    which is then used.
    
    Parameters
    ----------
    ID : str
        ID string of the target
    download_dir : str
        Directory for fits file and search results caches. 
    lkwargs : dict
        Dictionary to be passed to LightKurve  
    
    Returns
    -------
    lcCol : Lightkurve.LightCurveCollection instance
        Contains a list of all the sectors/quarters of data either freshly 
        downloaded or from the cache.
    """
    
    ID = format_name(ID)

    set_mission(ID, lkwargs)
    
    ID = getMASTidentifier(ID, lkwargs)

    search = check_sr_cache(ID, lkwargs, use_cached, download_dir=download_dir, cache_expire=cache_expire)
    
    fitsFiles = check_fits_cache(search, lkwargs['mission'], download_dir=download_dir)

    lcCol = load_fits(fitsFiles, lkwargs['mission'])
    
    return lcCol