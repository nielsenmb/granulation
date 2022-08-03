import lightkurve as lk
from lightkurve.periodogram import Periodogram
import jax.numpy as jnp
import numpy as np
from astropy.timeseries import LombScargle
from astropy import units
from scipy.integrate import simps
from matplotlib.pyplot import *

class psd():
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
                 downloadDir='./',fit_mean=False, timeConversion=86400):

        self.ID = ID
        self.downloadDir= downloadDir

        if (time is None) and (flux is None):
            
            time, flux = self._getTS(numaxGuess)

            flux = (flux/np.nanmedian(flux) - 1) * 1e6
                
        self.time = time

        self.flux = flux

        self.flux_err = flux_err

        self.fit_mean = fit_mean

        self.timeConversion = timeConversion

        self._getBadIndex()
 
        self.dt = self._getSampling()

        self.dT = self.time[-1] - self.time[0]

        self.NT = len(self.time)

        self.dutyCycle = self._getDutyCycle()

        if flux_err is None:
            # Init Astropy LS class without weights
            self.ls = LombScargle(self.time[self.indx] * self.timeConversion,
                                  self.flux[self.indx], center_data=True,
                                  fit_mean=self.fit_mean)

        else:
            # Init Astropy LS class with weights
            self.ls = LombScargle(self.time[self.indx] * self.timeConversion,
                                  self.flux[self.indx], center_data=True,
                                  fit_mean=self.fit_mean,
                                  dy=self.flux_err[self.indx],)

        self.Nyquist = 1/(2*self.timeConversion*self.dt) # Hz

        self.df = self._fundamental_spacing_integral()
 
    def _getTS(self, numaxGuess, exptime=None):
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

        if 'KIC' in self.ID:
            author = 'Kepler'

            mission = 'Kepler'

            if exptime is None:
                if numaxGuess[0] > 1/(2*1800)*1e6:
                    exptime = 60
                else:
                    exptime = 1800

        if 'TIC' in self.ID:
            author = 'SPOC'

            mission = 'TESS'
            
            if exptime is None:
                exptime = 120

        # Set window length in terms of bins depending on cadence
        wlen = int(1.5e6/exptime)-1

        # Window length must be odd
        if wlen % 2 == 0:
            wlen += 1
         
        LCcol = lk.search_lightcurve(self.ID, author=author, mission=mission, exptime=exptime).download_all(download_dir=self.downloadDir)
 
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

        pg = Periodogram(self.freq * units.uHz, units.Quantity(self.powerdensity))

        pg = pg.flatten()
         
        self.snr = pg.power.value

    def _getBadIndex(self):
        """ Identify indices with nan/inf values

        Flags array indices where either the timestamps, flux values, or flux errors
        are nan or inf.

        """

        if self.flux_err is not None:
            self.indx = np.invert(np.isnan(self.time) | np.isnan(self.flux) | np.isnan(self.flux_err) | np.isinf(self.time) | np.isinf(self.flux) | np.isinf(self.flux_err))
        else:
            self.indx = np.invert(np.isnan(self.time) | np.isnan(self.flux) | np.isinf(self.time) | np.isinf(self.flux))

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
        df = 1/(self.timeConversion*(np.nanmax(self.time[self.indx]) - np.nanmin(self.time[self.indx]))) # Hz

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