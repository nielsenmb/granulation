from lightkurve.periodogram import Periodogram
import jax.numpy as jnp
import numpy as np
from astropy.timeseries import LombScargle
from astropy import units
from scipy.integrate import simps
from utils import scalingRelations
import os, pickle, re
import lightkurve as lk
from datetime import datetime
 
class psd(scalingRelations):
    """ Asteroseismology wrapper for Astropy Lomb-Scargle

    Uses the Astropy.LombScargle class to compute the power spectrum of a given
    time series. A variety of choices for computing the spectrum are available.
    The recommended methods are either `fast' or `Cython'.

    The variable list includes Parameters which are input parameters and
    Attributes, which are class attributes set when the class instance is
    either initalized or called.

    Notes
    -----
    The Cython implemenation is very slow for time series longer than about
    1 month (array size of ~1e5). The Fast implementation is similar to the an
    FFT, but at a very slight loss of accuracy.

    The adjustments to the frequency resolution, due to gaps, performed in the
    KASOC filter may not be beneficial the statistics we use in the detection
    algorithm.  This has not been thuroughly tested yet though. So recommend
    leaving it in, but with a switch to turn it off for testing.

    Parameters
    ----------
    time : array
        Time stamps of the time series.
    flux : array
        Flux values of the time series.
    flux_error : array
        Flux value errors of the time series.
    fit_mean : bool, optional
        Keyword for Astropy.LombScargle. If True, uses the generalized
        Lomb-Scargle approach and fits with a floating mean. Default is False.
    timeConversion : float
        Factor to convert the time series such that it is in seconds. Note, all
        stored time values, e.g. cadence or duration, are kept in the input
        units. Default is 86400 to convert from days to seconds.

    Attributes
    ----------
    dt : float
        Cadence of the time series.
    dT : float
        Total length of the time series.
    NT : int
        Number of data points in the time series.
    dutyCycle : float
        Duty cycle of the time series.
    nyquist : float
        Nyquist frequency in Hz.
    df : float
        Fundamental frequency spacing in Hz.
    ls : astropy.timeseries.LombScargle object:
        Astropy Lomb-Scargle class instance used in computing the power
        spectrum.
    indx : array, bool
        Mask array for removing nan and/or -inf values from the time series.
    freqHz : array, float
        Frequency range in Hz.
    freq : array, float
        Freqeuency range in muHz.
    normfactor : float
        Normalization factor to ensure the power conforms with Parseval.
    power : array, float
        Power spectrum of the time series in ppm^2.
    powerdensity : array float
        Power density spectrum of the time series in ppm^2/muHz
    amplitude : array, float
        Amplitude spectrum of the time series in ppm.
    """

    def __init__(self, ID, time=None, flux=None, flux_err=None, numaxGuess=None, 
                 downloadDir='./',fit_mean=False, timeConversion=86400, tBinsMax=3, pBinsMax=10):

        self.ID = ID

        self.downloadDir= downloadDir

        if (time is None) and (flux is None):
            
            time, flux = self._getTS(numaxGuess)

            flux = (flux/np.nanmedian(flux) - 1) * 1e6
         
        self._time = time
        
        self._flux = flux

        self._getBadIndex(time, flux)

        self.pbins, self.tbins = self.determineBins(time[self.indx], numaxGuess, tBinsMax=tBinsMax, pBinsMax=pBinsMax)
 
        if self.tbins > 1:  
            print(f'Binning time series by {self.tbins}')
            self.time, self.flux = self.binning(time[self.indx], flux[self.indx], self.tbins)
        else:
            self.time, self.flux = time[self.indx], flux[self.indx]
 
        self.fit_mean = fit_mean

        self.timeConversion = timeConversion

        self.dt = self._getSampling()

        self.dT = self.time.max() - self.time.min()

        self.NT = len(self.time)

        self.dutyCycle = self._getDutyCycle()

        if flux_err is None:
            # Init Astropy LS class without weights
            self.ls = LombScargle(self.time * self.timeConversion,
                                  self.flux, center_data=True,
                                  fit_mean=self.fit_mean)

        else:
            # Init Astropy LS class with weights
            self.ls = LombScargle(self.time * self.timeConversion,
                                  self.flux, center_data=True,
                                  fit_mean=self.fit_mean,
                                  dy=self.flux_err,)

        self.Nyquist = 1/(2*self.timeConversion*self.dt) # Hz

        self.df = self._fundamental_spacing_integral()

    def determineBins(self, t, numaxGuess, tgtLen=1e5, nW=2, tBinsMax=3, pBinsMax=10):

        df = 1/((t[-1] - t[0])*24*60*60)*1e6

        nyq = 1/(2*np.median(np.diff(t*24*60*60)))*1e6

        if (numaxGuess is None) or nyq < 1000:
            nT = 1
        else:
            nT = min([max([nyq//(numaxGuess + nW*self.envWidth(numaxGuess)), 1]), tBinsMax])

        nP = 1

        while (nyq//nT-df*nP)//(df*nP) > tgtLen:
            nP += 1

        if nP > pBinsMax:
            nP = pBinsMax

        return int(nP), int(nT)

    def binning(self, _t, _d, n):
        """
        Binning is done in chunks.

        Parameters
        ----------
        _t : _type_
            _description_
        _d : _type_
            _description_
        n : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        N = len(_t)

        t, d = np.array([]), np.array([])

        Nc = 100000

        for i in range(N//Nc+1):
             
            _x , _y = self._binTS(_t[i*Nc: (i+1)*Nc], _d[i*Nc: (i+1)*Nc], n)

            t = np.append(t, _x)

            d = np.append(d, _y)

        return t, d
     
    def _binTS(self, t, d, n):

        dt = np.median(np.diff(t))

        T = np.zeros_like(t)  

        D = np.zeros_like(d)  

        i, j = 0, 0
 
        while i < len(t)-n+1:
            
            dT = t[i+n-1] - t[i]

            if dT < (n + 1) * dt:
                
                T[j] = t[i + n - 1] - dT / 2
 
                D[j] = np.mean(d[i:i+n])
 
                i += n

                j += 1
            else:
                i +=1
                
        return T[:j], D[:j]

    def _getTS(self, numaxGuess):
        """Get time series with lightkurve

        Parameters
        ----------
        exptime : int, optional
            The exposure time (cadence) of the data to get, by default None.

        Returns
        -------
        time : DeviceArray
            The timestamps of the time series.
        flux : DeviceArray
            The flux values of the time series.
        """

        lk_kwargs = {}

        if 'KIC' in self.ID:
            lk_kwargs['author'] = 'Kepler'
            lk_kwargs['mission'] = 'Kepler'

            if numaxGuess > 1/(2*1800)*1e6:
                lk_kwargs['exptime'] = 60
                self.lc_type = 'Kepler 60s'
            else:
                lk_kwargs['exptime'] = 1800
                self.lc_type = 'Kepler 1800s'

        if 'TIC' in self.ID:
            lk_kwargs['author'] = 'SPOC'
            lk_kwargs['mission'] = 'TESS'
            lk_kwargs['exptime'] = 120
            self.lc_type = 'TESS 120s'

        wlen = int(1.5e6/lk_kwargs['exptime'])-1
        if wlen % 2 == 0:
            wlen += 1
        
        try:
            LCcol = search_lightcurve(self.ID, self.downloadDir, lk_kwargs, use_cached=True, cache_expire=10*365)  
        except:
            LCcol = search_lightcurve(self.ID, self.downloadDir, lk_kwargs, use_cached=True, cache_expire=0)
              
        lc = LCcol.stitch().normalize().remove_nans().remove_outliers().flatten(window_length=wlen)

        t, d = jnp.array(lc.time.value), jnp.array(lc.flux.value)
 
        return t, d

    def __call__(self, oversampling=1, nyquist_factor=1.0, method='fast'):
        """ Compute power spectrum

        Computes the power spectrum and normalizes it to conform with Parseval's
        theorem. The output is available as the power in ppm^2, powerdensity in
        ppm^2/muHz and the amplitude spectrum in ppm.

        The frequency range is transformed to muHz as this is customarily used
        in asteroseismology of main sequence stars.

        Parameters
        ----------
        oversampling : int
            The number of times the frequency range should be oversampled. This
            equates to zero-padding when using the FFT.
        nyquist_factor : float
            Factor by which to extend the spectrum past the Nyquist frequency.
            The default is 10% greater than the true Nyquist frequency. We use
            this to get a better handle on the background level at high
            frequency.
        method : str
            The recommended methods are either `fast' or `Cython'. Cython is
            a bit more accurate, but significantly slower.
        """

        self.freqHz = np.arange(self.df/oversampling, nyquist_factor*self.Nyquist, 
                                self.df/oversampling, dtype='float64')

        self.freq = jnp.array(self.freqHz*1e6) # muHz is usually used in seismology

        # Calculate power at frequencies using fast Lomb-Scargle periodogram:
        power = self.ls.power(self.freqHz, normalization='psd', method=method, 
                              assume_regular_frequency=True)

        # Due to numerical errors, the "fast implementation" can return power < 0.
        # Replace with random exponential values instead of 0?
        power = np.clip(power, 0, None)

        self._getNorm(power)

        self.power = jnp.array(power * self.normfactor * 2)

        self.powerdensity = jnp.array(power * self.normfactor / (self.df * 1e6))
 
        self.amplitude = jnp.array(power * np.sqrt(power * self.normfactor * 2))

        if self.pbins > 1:
            print(f'Binning PSD by {self.pbins}')
            self.freq = self.binPSD(self.freq)
            self.power = self.binPSD(self.power)
            self.powerdensity = self.binPSD(self.powerdensity)
            self.amplitude = self.binPSD(self.amplitude)

        pg = Periodogram(self.freq * units.uHz, units.Quantity(self.powerdensity))

        pg = pg.flatten()
         
        self.snr = pg.power.value


    def binPSD(self, inp):
        """ Bin x by a factor n

        If len(x) is not equal to an integer number of n, the remaining
        frequency bins are discarded. Half at low frequency and half at high
        frequency.

        Parameters
        ----------
        inp : array
            Array of values to bin.

        Returns
        -------
        xbin : array
            The binned version of the input array
        """

        x = inp.copy()

        # Number of frequency bins per requested bin width
        n = self.pbins

        # The input array isn't always an integer number of the binning factor
        # A bit of the input array is therefore trimmed a low and high end.
        trim = (len(x)//n)*n

        half_rest = (len(x)-trim)//2

        x = x[half_rest:half_rest+trim] # Trim the input array

        xbin = x.reshape((-1, n)).mean(axis = 1) # reshape and average

        return xbin

    def _getBadIndex(self, time, flux):
        """ Identify indices with nan/inf values

        Flags array indices where either the timestamps, flux values, or flux errors
        are nan or inf.

        """

        self.indx = np.invert(np.isnan(time) | np.isnan(flux) | np.isinf(time) | np.isinf(flux))

    def getTSWindowFunction(self, tmin=None, tmax=None, cadenceMargin=1.01):

        if tmin is None:
            tmin = min(self.time)

        if tmax is None:
            tmax = max(self.time)

        t = self.time.copy()[self.indx]

        w = np.ones_like(t)

        break_counter = 0
        epsilon = 0.0001 # this is a tiny scaling of dt to avoid numerical issues

        while any(np.diff(t) > cadenceMargin*self.dt):

            idx = np.where(np.diff(t)>cadenceMargin*self.dt)[0][0]

            t_gap_fill = np.arange(t[idx], t[idx+1]-epsilon*self.dt, self.dt)

            w_gap_fill = np.zeros(len(t_gap_fill))
            w_gap_fill[0] = 1

            t = np.concatenate((t[:idx], t_gap_fill, t[idx+1:]))

            w = np.concatenate((w[:idx], w_gap_fill, w[idx+1:]))

            break_counter +=1
            if break_counter == 100:
                break

        if (tmin is not None) and (tmin < t[0]):
            padLow = np.arange(tmin, t[0], self.dt)
            t = np.append(padLow, t)
            w = np.append(np.zeros_like(padLow), w)

        if (tmax is not None) and (t[0] < tmax):
            padHi = np.arange(t[-1], tmax, self.dt)
            t = np.append(t, padHi)
            w = np.append(w, np.zeros_like(padHi))

        return t, w

    def _getDutyCycle(self, cadence=None):
        """ Compute the duty cycle

        If cadence is not provided, it is assumed to be the median difference
        of the time stamps in the time series.

        Parameters
        ----------
        cadence : float
            Nominal cadence of the time series. Units should be the
            same as t.

        Returns
        -------
        dutyCycle : float
            Duty cycle of the time series
        """

        if cadence is None:
            cadence = self._getSampling()

        nomLen = np.ceil((np.nanmax(self.time) - np.nanmin(self.time)) / cadence)

        idx = np.invert(np.isnan(self.time) | np.isinf(self.time))

        dutyCycle = len(self.time[idx]) / nomLen

        return dutyCycle

    def _getSampling(self):
        """ Compute sampling rate

        Computes the average sampling rate in the time series.

        This should approximate the nominal sampling rate,
        even with gaps in the time series.

        Returns
        ----------
        dt : float
            Cadence of the time stamps.
        """
        idx = np.invert(np.isnan(self.time) | np.isinf(self.time))

        dt = np.median(np.diff(self.time[idx]))

        return dt

    def _getNorm(self, power):
        """ Parseval normalization

        Computes the normalization factor for the power spectrum such that it
        conforms with Parseval's theorem.

        power : array
            Unnormalized array of power.
        """

        N = len(self.ls.t)

        if self.ls.dy is None:
            tot_MS = np.sum((self.ls.y - np.nanmean(self.ls.y))**2)/N
        else:
            tot_MS = np.sum(((self.ls.y - np.nanmean(self.ls.y))/self.ls.dy)**2)/np.sum((1/self.ls.dy)**2)

        self.normfactor = tot_MS/np.sum(power)

    def _fundamental_spacing_integral(self):
        """ Estimate fundamental frequency bin spacing

        Computes the frequency bin spacing using the integral of the spectral
        window function.

        For uniformly sampled data this is given by df=1/T. Which under ideal
        circumstances ensures that power in neighbouring frequency bins is
        independent. However, this fails when there are gaps in the time series.
        The integral of the spectral window function is a better approximation
        for ensuring the bins are less correlated.

        """

        # The nominal frequency resolution
        df = 1/(self.timeConversion*(np.nanmax(self.time) - np.nanmin(self.time))) # Hz

        # Compute the window function
        freq, window = self.windowfunction(df, width=100*df, oversampling=5) # oversampling for integral accuracy

        # Integrate the windowfunction to get the corrected frequency resolution
        df = simps(window, freq)

        return df*1e-6

    def windowfunction(self, df, width=None, oversampling=10):
        """ Spectral window function.

        Parameters
        ----------
		 width : float, optional
            The width in Hz on either side of zero to calculate spectral window.
            Default is None.
        oversampling : float, optional
            Oversampling factor. Default is 10.
        """

        if width is None:
            width = 100*df

        freq_cen = 0.5*self.Nyquist

        Nfreq = int(oversampling*width/df)

        freq = freq_cen + (df/oversampling) * np.arange(-Nfreq, Nfreq, 1)

        x = 0.5*np.sin(2*np.pi*freq_cen*self.ls.t) + 0.5*np.cos(2*np.pi*freq_cen*self.ls.t)

        # Calculate power spectrum for the given frequency range:
        ls = LombScargle(self.ls.t, x, center_data=True, fit_mean=self.fit_mean)

        power = ls.power(freq, method='fast', normalization='psd', assume_regular_frequency=True)

        power /= power[int(len(power)/2)] # Normalize to have maximum of one

        freq -= freq_cen

        freq *= 1e6

        return freq, power


def squish(time, dt, gapSize=27):
    """ Remove gaps

    Adjusts timestamps to remove gaps of a given size. Large gaps influence
    the statistics we use for the detection quite strongly.

    Parameters
    ----------
    gapSize : float
        Size of the gaps to consider, in units of the timestamps. Gaps
        larger than this will be removed. Default is 27 days.

    Returns
    -------
    t : array
        Adjusted timestamps
    """

    tsquish = time.copy()

    for i in np.where(np.diff(tsquish) > gapSize)[0]:
        diff = tsquish[i] - tsquish[i+1]

        tsquish[i+1:] = tsquish[i+1:] + diff + dt

    return tsquish




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
    elif mission in ['TESS']:
        lcs = [lk.lightcurvefile.TessLightCurveFile(file) for file in files if os.path.basename(file).startswith('tess')]
        lcCol = lk.LightCurveCollection(lcs)
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

    search = lk.search_lightcurve(ID, 
                                  exptime=lkwargs['exptime'], 
                                  mission=lkwargs['mission'], 
                                  author=lkwargs['author'])
     
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
        
        search = lk.search_lightcurvefile(ID, exptime=lkwargs['exptime'], 
                                          mission=lkwargs['mission'], 
                                          author=lkwargs['author'])

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