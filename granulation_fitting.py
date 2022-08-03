import jax.numpy as jnp
from functools import partial
import jax, dynesty, corner, os
from dynesty import utils as dyfunc
import distributions as dist
import numpy as np
import lightkurve as lk
import lightkurveCacheAccess as lka
jax.config.update('jax_enable_x64', True)

class scalingRelations():
    """ Container for scaling relations

    This is a helper class which contains methods for the various scaling
    relations.

    """

    def __init_(self):
        pass

    @partial(jax.jit, static_argnums=(0,))
    def envWidth(self, numax):
        """ Scaling relation for the envelope width

        Computest he full width at half maximum of the p-mode envelope based
        on numax and Teff (optional).

        Parameters
        ----------
        numax : float
            Frequency of maximum power of the p-mode envelope.
        Teff : float, optional
            Effective surface temperature of the star.
        Teff0 : float, optional
            Solar effective temperature in K. Default is 5777 K.

        Returns
        -------
        width : float
            Envelope width in muHz
        """

        width = 0.66*numax**0.88 # Mosser et al. 201??

        return width

    @partial(jax.jit, static_argnums=(0,))
    def nuHarveyGran(self, numax):
        """ Harvey frequency for granulation term

        Scaling relation for the characteristic frequency of the granulation
        noise. Based on Kallinger et al. (2014).

        Parameters
        ----------
        numax : float
            Frequency of maximum power of the p-mode envelope.

        Returns
        -------
        nu : float
            Characteristic frequency of Harvey law for granulation.

        """

        nu = 0.317 * numax**0.970

        return nu

    @partial(jax.jit, static_argnums=(0,))
    def nuHarveyEnv(self, numax):
        """ Harvey frequency for envelope term

        Scaling relation for the characteristic frequency of the envelope
        noise. Based on Kallinger et al. (2014).

        Parameters
        ----------
        numax : float
            Frequency of maximum power of the p-mode envelope.

        Returns
        -------
        nu : float
            Characteristic frequency of Harvey law for envelope.

        """

        nu = 0.948 * numax**0.992

        return nu

class granulation_fit(scalingRelations):
 
    def __init__(self, ID, numax, download_dir):


        self.ID = ID

        self.numax_guess = numax

        self.download_dir = download_dir

        f, p = self.getPSD()

        self.f = jnp.array(f)

        self.p = jnp.array(p)

        self.nyq = self.f.max()

        self.ndim = 13

        self.eta = jnp.sinc(self.f / 2.0 / self.nyq)**2.0

        self.addPriors()

        self.wfac = 1/ (2 * jnp.sqrt(2 * jnp.log(2))) / 2

        if self.pbins > 1:
            self._lnlike = self._lnlike_binned
        else:
            self._lnlike = self._lnlike_std

    def getPSD(self,):


        lk_kwargs = {}
        if 'KIC' in self.ID:
            lk_kwargs['author'] = 'Kepler'
            lk_kwargs['mission'] = 'Kepler'

            if self.numax_guess[0] > 1/(2*1800)*1e6:
                lk_kwargs['exptime'] = 60
            else:
                lk_kwargs['exptime'] = 1800

        if 'TIC' in self.ID:
            lk_kwargs['author'] = 'SPOC'
            lk_kwargs['mission'] = 'TESS'
            lk_kwargs['exptime'] = 120

        wlen = int(1.5e6/lk_kwargs['exptime'])-1
        if wlen % 2 == 0:
            wlen += 1

         
        #LCcol = lkA.search_lightcurve(self.ID, author=author, mission=mission, exptime=exptime).download_all(download_dir=self.download_dir)
        LCcol = lka.search_lightcurve(self.ID, self.download_dir, lk_kwargs, use_cached=True, cache_expire=1)
        
        lc = LCcol.stitch().normalize().remove_nans().remove_outliers().flatten(window_length=wlen) 

        self.pbins, self.tbins = self.determineBins(lc.time.value)

        if self.tbins > 1: #((exptime == 60) or (exptime == 120)) and (self.numax_guess[0] < 1/(2*exptime)*1e6/self.tbins):
            print(f'Binning time series by {self.tbins}')

            self.t, self.d = binning(lc.time.value, lc.flux.value, self.tbins)

            lc = lk.LightCurve(time=self.t, flux=self.d, targetid=self.ID)

        else:
            self.t, self.d = lc.time.value, lc.flux.value

        pg = lc.to_periodogram(normalization='psd', minimum_frequency=1.0)

        f, p = pg.frequency.value, pg.power.value

        tot_MS = np.nansum((self.d - np.nanmean(self.d))**2)/len(self.t)

        normfactor = tot_MS/np.sum(p)

        p *= normfactor

        if self.pbins > 1:
            print(f'Binning PSD by {self.pbins}')
            f = self.binPSD(f)

            p = self.binPSD(p)

        return f, p

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

    def determineBins(self, t, tgtLen=1e5, nW=2, tBinMax=3, pBinMax=10):

        df = 1/((t[-1] - t[0])*24*60*60)*1e6

        nyq = 1/(2*np.median(np.diff(t*24*60*60)))*1e6

        nT = min([max([nyq//(self.numax_guess[0] + nW*self.envWidth(self.numax_guess[0])), 1]), tBinMax])

        if nyq < 1000:
            nT = 1

        nP = 1

        while (nyq//nT-df*nP)//(df*nP) > tgtLen:
            nP += 1

        if nP > pBinMax:
            nP = pBinMax

        return int(nP), int(nT)

    def addPriors(self):
        self.priors = []

        widx = self.f > self.nyq - 100
        pw = self.p[widx].mean()

        # Harvey 1 (envelope harvey)
        self.addHarveyPriors(self.nuHarveyEnv(self.numax_guess[0]), pw)

        # Harvey 2 (granulation)
        self.addHarveyPriors(self.nuHarveyGran(self.numax_guess[0]), pw)

        # Harvey 3 Activity
        self.addHarveyPriors(self.f[0], pw)

        # numax
        self.priors.append(dist.normal(mu=self.numax_guess[0], sigma=self.numax_guess[1]))

        # Envelope dwidth
        self.priors.append(dist.beta(a=1.2, b=1.2, loc=0.2, scale=2))

        # Envelope height
        numax_idx = (abs(self.f - self.numax_guess[0]) < self.envWidth(self.numax_guess[0]))
        self.priors.append(dist.normal(mu=jnp.log10(self.p[numax_idx].mean()), sigma=2))

        # White noise
        self.priors.append(dist.normal(mu=jnp.log10(pw).mean(), sigma=2))

        self.labels = ['hsig1', 'dhnu1', 'hexp1',
                       'hsig2', 'dhnu2', 'hexp2',
                       'hsig3', 'hnu3', 'hexp3',
                       'numax', 'dwidth', 'height',
                       'white']

    def addHarveyPriors(self, nuH, pw):
        """ Append Harvey law prior

        Adds the priors for the three Harvey law parametets hsig, dhnu, and
        hexp.

        hsig is proportional to the harvey law amplitude in ppm^2. The prior
        is a normal distribution in log(amplitude) which determined based on
        the observed power in the spectrum (this is bad Baysian I know).

        dhnu is a scaling factor for the scaling relation for the relevant
        Harvey law frequency (from Kallinger et al. (2014) and the prior is
        a beta distribution between 0.2 and 3.

        hexp is a the exponent in the Harvey law and the prior is a beta
        distribution which ranges between 1 and 6.

        Each prior is appended to the complete list of priors used in the
        sampling.

        Parameters
        ----------
        nuH : float
            Characteristic for the relevant Harvey term
        pw : float
            White noise level.

        """

        idx = abs(self.f - nuH) < 10

        mu = jnp.array([1, self.p[idx].mean() - pw]).max()

        self.priors.append(dist.normal(mu=jnp.log10(mu*nuH), sigma=1)) # hsig

        self.priors.append(dist.beta(a=1.2, b=1.2, loc=0.2, scale=3)) # dhnu

        self.priors.append(dist.beta(a=1.2, b=1.2, loc=1, scale=5)) # hexp

    @partial(jax.jit, static_argnums=(0,))
    def ptform(self, u):

        x = [self.priors[n].ppf(u[n]) for n in range(self.ndim)]

        return x

    @partial(jax.jit, static_argnums=(0,))
    def harvey(self, hsig, hnu, hexp):
        """
        Harvey profile
        """

        har = 10**hsig / hnu / (1.0 + (self.f / hnu)**hexp)

        return har

    @partial(jax.jit, static_argnums=(0,))
    def gaussian(self, numax, width, height):
        """
        Gaussian to describe oscillations
        """

        m = 10**height * jnp.exp(-(self.f - numax)**2 / (2.0 * width**2))

        return m

    @partial(jax.jit, static_argnums=(0,))
    def model(self, params):

        hsig1, dhnu1, exp1, hsig2, dhnu2, exp2, hsig3, hnu3, exp3, numax, dwidth, height, white = params

        # Start with Harvey laws
        hnu1 = self.nuHarveyEnv(numax) * dhnu1
        H1 = self.harvey(hsig1, hnu1, exp1)

        hnu2 = self.nuHarveyGran(numax) * dhnu2
        H2 = self.harvey(hsig2, hnu2, exp2)

        H3 = self.harvey(hsig3, hnu3, exp3)

        # Add envelope
        width = dwidth * self.envWidth(numax) * self.wfac
        G = self.gaussian(numax, width, height)

        # Add white noise
        W = 10**white

        model = (H1 + H2 + H3 + G)*self.eta + W

        return model, {'H1': H1, 'H2': H2, 'H3': H3, 'G': G, 'W': W}

    @partial(jax.jit, static_argnums=(0,))
    def lnfactorial(self, n):
        """ log(n!) approximation

        For large n the scipy/numpy implimentations croak when doing factorials.

        We therefore use the Ramanujan approximation to compute log(n!).

        Parameters:
        -----------
        n : int
            Value to compute the factorial of.

        Returns
        -------
        r : float
            The approximate value of log(n!)
        """

        r = n * jnp.log(n) - n + jnp.log(n*(1+4*n*(1+2*n)))/6 + jnp.log(jnp.pi)/2

        return r

    @partial(jax.jit, static_argnums=(0,))
    def _lnlike_std(self, mod):
        L = -jnp.sum(jnp.log(mod) + self.p/mod)
        return L

    @partial(jax.jit, static_argnums=(0,))
    def _lnlike_binned(self, mod):
        L = jnp.sum((self.pbins-1)*(jnp.log(self.pbins) + jnp.log(self.p)) - self.lnfactorial(self.pbins-1) - self.pbins*(jnp.log(mod) + self.p/mod))
        return L

    @partial(jax.jit, static_argnums=(0,))
    def lnlike(self, params):

        mod, _ = self.model(params)

        hsig1, dhnu1, exp1, hsig2, dhnu2, exp2, hsig3, hnu3, exp3, numax, dwidth, height, white = params

        hnu1 = self.nuHarveyEnv(numax) * dhnu1

        hnu2 = self.nuHarveyGran(numax) * dhnu2

        T = (hnu3 > hnu2) & \
            (hnu2 > hnu1) & \
            (hnu2 > numax) & \
            (hnu1/hnu2 < 0.1)

        L = jax.lax.cond(T, lambda mod: -jnp.inf, self._lnlike, mod)

        return L

    def runDynesty(self, nlive=200):

        sampler = dynesty.NestedSampler(self.lnlike, self.ptform, self.ndim, nlive=nlive)

        sampler.run_nested(print_progress=False)

        result = sampler.results

        samples, weights = result.samples, jnp.exp(result.logwt - result.logz[-1])

        new_samples = dyfunc.resample_equal(samples, weights)

        return sampler, new_samples

    def makeCorner(self, fig, samples, N, labels=None, outputDir=None):

        thinIdx = np.random.choice(np.arange(samples.shape[0]), size=min([N, samples.shape[0]]), replace=False)

        ndim = len(labels)

        idx = np.sum(np.array([x == np.array(self.labels) for x in labels]), axis = 0).astype(bool)

        corner.corner(samples[thinIdx, :][:, idx], labels=labels, hist_kwargs={'density': True}, fig=fig);

        pr = np.array(self.priors)[idx]

        axes = np.array(fig.axes).reshape((ndim, ndim))

        for i in range(ndim):
            xl = jnp.linspace(pr[i].ppf(0.001), pr[i].ppf(0.5), 101)

            xu = jnp.linspace(pr[i].ppf(0.5), pr[i].ppf(0.999), 101)

            axes[i,i].plot(xl, pr[i].pdf(xl), color = 'C0')

            axes[i,i].plot(xu, pr[i].pdf(xu), color = 'C1')

            axes[i,i].axvline(pr[i].ppf(0.5), color = 'k', ls='dashed')

        if not outputDir is None:
            if len(labels) == self.ndim:
                fname = '_all.png'

            else:
                fname = '_'+'_'.join(labels)+'.png'

            path = os.path.join(*[outputDir, os.path.basename(outputDir) + fname])

            fig.savefig(path)

        #return fig, axes

    def plotModel(self, fig, ax, samples=None, outputDir=None):

        def plotLines(ax, params, idx=None, alpha=1):

            if idx is None:
                idx = np.arange(len(self.f))

            model, comps = self.model(params)

            ax.plot(self.f[idx], model[idx], lw =5, color = 'k', alpha=alpha)

            ax.plot(self.f[idx], comps['H1'][idx], ls = 'dashed', lw = 3, color = 'C1', alpha=alpha)

            ax.plot(self.f[idx], comps['H2'][idx], ls = 'dashed', lw = 3, color = 'C2', alpha=alpha)

            ax.plot(self.f[idx], comps['H3'][idx], ls = 'dashed', lw = 3, color = 'C3', alpha=alpha)

            ax.plot(self.f[idx], comps['G'][idx], ls = 'dashed', lw = 3, color = 'C4', alpha=alpha)

            ax.axhline(comps['W'], color = 'k', ls='dashed', alpha=alpha)

        ax.plot(self.f, self.p, alpha = 0.5, color = 'C0')

        if samples is None:
            mu_pr = np.zeros(self.ndim) + 0.5

            params = self.ptform(mu_pr)

            plotLines(ax, params)

        else:
            nsamples = len(samples[:, 0])

            thinIdx = gen_log_space(len(self.f), 300)

            rnd = np.random.choice(np.arange(nsamples), size=30, replace=False)

            for idx in rnd:
                plotLines(ax, samples[idx, :], thinIdx, alpha = 0.1)

        ax.set_ylim(max([self.p.min()*0.9]), self.p.max()*1.1)

        ax.set_xlim(self.f.min(), self.f.max())

        ax.set_yscale('log')

        ax.set_xscale('log')

        ax.set_ylabel('Power spectral density [ppm$^2/\mu$Hz]')

        ax.set_xlabel('Frequency [$\mu$Hz]')

        if outputDir is not None:

            if samples is None:
                path = os.path.join(*[outputDir, os.path.basename(outputDir) + '_prior_model.png'])
            else:
                path = os.path.join(*[outputDir, os.path.basename(outputDir) + '_samples_model.png'])

            fig.savefig(path, dpi=300)

    def storeResults(self, i, updated_data, new_keys, samples, outputDir):

        hsig1, dhnu1, exp1, hsig2, dhnu2, exp2, hsig3, hnu3, exp3, numax, dwidth, height, white = samples.T

        hnu1 = self.nuHarveyEnv(numax) * dhnu1

        hnu2 = self.nuHarveyGran(numax) * dhnu2

        width = dwidth * self.envWidth(numax) / (2 * jnp.sqrt(2 * jnp.log(2)))

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

        updated_data['completed'] = 1

        if outputDir is not None:

            path = os.path.join(*[outputDir, os.path.basename(outputDir) + '_samples'])

            np.savez_compressed(path, samples=samples)

def _binTS(t, d, n):

    dt = np.median(np.diff(t))

    T = np.array([])

    D = np.array([])

    i = 0

    while i < len(t)-n+1:

        dT = t[i+n-1]-t[i]

        if dT < (n+1)*dt:
            T = np.append(T, t[i+n-1]-dT/2)

            D = np.append(D, np.mean(d[i:i+n]))

            i += n

        else:
            i +=1

    return T, D

def binning(_t, _d, n):

    N = len(_t)

    t, d = np.array([]), np.array([])

    Nc = 200000

    for i in range(N//Nc+1):

        _x , _y = _binTS(_t[i*Nc: (i+1)*Nc], _d[i*Nc: (i+1)*Nc], n)

        t = np.append(t, _x)

        d = np.append(d, _y)

    return t, d

def gen_log_space(limit, n):

    result = [1]

    if n>1:  # just a check to avoid ZeroDivisionError
        ratio = (float(limit)/result[-1]) ** (1.0/(n-len(result)))

    while len(result)<n:
        next_value = result[-1]*ratio

        if next_value - result[-1] >= 1:
            # safe zone. next_value will be a different integer
            result.append(next_value)

        else:
            # problem! same integer. we need to find next_value by artificially incrementing previous value
            result.append(result[-1]+1)

            # recalculate the ratio so that the remaining values will scale correctly
            ratio = (float(limit)/result[-1]) ** (1.0/(n-len(result)))

    # round, re-adjust to 0 indexing (i.e. minus 1) and return np.uint64 array
    return np.array(list(map(lambda x: round(x)-1, result)), dtype=np.uint64)