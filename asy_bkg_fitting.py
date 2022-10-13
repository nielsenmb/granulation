from curses.has_key import has_key
import jax.numpy as jnp
from functools import partial
import jax, dynesty, corner, os, dill, time, emcee, utils
from dynesty import utils as dyfunc
import numpy as np
from IO import psd
from utils import scalingRelations
from asymptotic import asymptotic
from DR import PCA
import astropy.convolution as conv
jax.config.update('jax_enable_x64', True)


class spectrum_fit(scalingRelations, asymptotic):
    
    def __init__(self, ID, obs, download_dir, pcadim=None, 
                 fname='PCAsample.csv', weights=None, weight_args={}, N=200, ):

        self.ID = ID
     
        self.download_dir = download_dir

        self.obs = obs

        self.logobs = {key: utils.to_log10(*obs[key]) for key in obs.keys()}
                
        self.getPSD()
        
        self.norders = 7

        self.V20 = 0.5

        # Envelope width conversion factor (only used when running without pca)
        self.wfac = 1 / (2 * jnp.sqrt(2 * jnp.log(2))) / 2

        # Pick the likelihood function
        if self.psd.pbins > 1:
            self._lnlike = self._lnlike_binned
        else:
            self._lnlike = self._lnlike_std

        # Labels for all the parameters
        self.labels = ['dnu', 'd02', 'alpha', 'mode_width', 'teff', 'bp_rp', 'eps',
                       'numax', 'env_height', 'env_width', 
                       'H1_power', 'H1_nu', 'H1_exp',
                       'H2_power', 'H2_nu', 'H2_exp',
                       'H3_power', 'H3_nu', 'H3_exp',
                       'white']

        if pcadim > 0:
            self.with_pca = True
        else:
            self.with_pca = False

        if self.with_pca:
            
            # Labels for parameters to be used in the pca
            self.pcalabels = ['dnu', 'd02', 'alpha', 'mode_width', 'teff', 'bp_rp', 'eps',
                              'numax', 'env_height', 'env_width',
                              'H1_power', 'H1_nu', 'H1_exp',
                              'H2_power', 'H2_nu', 'H2_exp',]

            self.ndim = pcadim + (len(self.labels) - len(self.pcalabels)) 

            self.initPCA(pcadim, weights, weight_args, N, fname)

        else:
            self.ndim = len(self.labels)

        self.addPriors()

        self.addObs()

    def getPSD(self):

        self.psd = psd(self.ID, numaxGuess=self.obs['numax'][0], downloadDir=self.download_dir, pBinsMax=1, tBinsMax=1)

        self.psd()

        self.f, self.p = self.psd.freq[self.psd.freq > 1], self.psd.powerdensity[self.psd.freq > 1]
        
        self._f = jnp.append(self.f, self.f + self.f.max())

        self.t, self.d = self.psd.time, self.psd.flux

        self.Nyquist = self.psd.Nyquist * 1e6

        self.eta = self.eta()

    def addObs(self,):
        self.obsLikes = []

        self.obsLikes.append(utils.normal(mu=self.obs['teff'][0], 
                                          sigma=self.obs['teff'][1]))

        self.obsLikes.append(utils.normal(mu=self.obs['bp_rp'][0], 
                                          sigma=self.obs['bp_rp'][1]))

    def addPriors(self):
        """Add a list of priors as an attribute to self

        The list of priors must appear in the label order (see below). 

        The list contains distribution class instances similar to the scipy.stats
        classes. These are faster though, and have been jaxed.

        """

        self.priors = []

        # Add the 3rd Harvey law and white noise
        widx = self.f > self.Nyquist - 10

        pw = np.nanmean(self.p[widx])

        if self.with_pca:
            # Add pca parameters to the list of prior objects
            for i in range(self.DR.dims_R):
                self.priors.append(utils.distribution(self.DR.ppf[i], 
                                                      self.DR.pdf[i], 
                                                      self.DR.logpdf[i], 
                                                      self.DR.cdf[i]))
        else:
             # numax
            self.priors.append(utils.normal(mu=self.obs['numax'][0], 
                                            sigma=self.obs['numax'][1]))

            # Envelope height
            numax_idx = (abs(self.f - self.obs['numax'][0]) < self.envWidth(self.obs['numax'][0]))
            self.priors.append(utils.normal(mu=jnp.log10(self.p[numax_idx].mean()), sigma=2))

            # Envelope dwidth
            self.priors.append(utils.beta(a=1.2, b=1.2, loc=0.2, scale=3))

            # Harvey 1 (envelope harvey)
            self.addHarveyPriors(self.nuHarveyEnv(self.obs['numax'][0]), pw)

            # Harvey 2 (granulation)
            self.addHarveyPriors(self.nuHarveyGran(self.obs['numax'][0]), pw)

        # Harvey 3 Activity
        self.addHarveyPriors(self.f[0], pw)
        self.priors[-2] = utils.beta(a=1.2, b=1.2, loc=0, scale=6) # replace H1_nu prior with a beta distribution

        # White noise
        self.priors.append(utils.normal(mu=jnp.log10(pw), sigma=2))
 
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

        self.priors.append(utils.normal(mu=jnp.log10(mu*nuH), sigma=2)) # hsig

        self.priors.append(utils.normal(mu=1, sigma=0.15)) # dhnu

        self.priors.append(utils.beta(a=1.2, b=1.2, loc=1.5, scale=3.5)) # hexp

    def initPCA(self, PCAdim, weights, weight_args, N, fname):
       
        _obs = self.logobs.copy()

        for key in ['bp_rp', 'eps']:
            if key in _obs.keys():
                _obs[key] = self.obs[key]
         
        self.DR = PCA(_obs, self.pcalabels, fname, N, 
                      weights=weights, weight_args=weight_args) 

        self.DR.fit_weightedPCA(PCAdim)

        _Y = self.DR.transform(self.DR.data_F)

        self.DR.ppf, self.DR.pdf, self.DR.logpdf, self.DR.cdf = self.DR.getQuantileFuncs(_Y)   

    def eta(self,):
        return jnp.sinc(self._f / 2.0 / self.Nyquist)**2.0    
    
    @partial(jax.jit, static_argnums=(0,))
    def ptform(self, u):
        """the prior transform function for the nested sampling

        Parameters
        ----------
        u : list
            List of floats between 0 and 1 with length equivalent to ndim. The 
            ppf of each prior distribution is evaluated based on this. 

        Returns
        -------
        x : list
            List of floats of the prior pdfs evaluated at each point in this
            list u.
        """
 
        x = jnp.array([self.priors[i].ppf(u[i]) for i in range(self.ndim)])

        return x

    @partial(jax.jit, static_argnums=(0,))
    def harvey(self, hsig, hnu, hexp):
        """ Standard Harvey profile

        Note the input hsig is log10.

        """

        har = hsig / hnu / (1.0 + (self._f / hnu)**hexp)

        return har

    @partial(jax.jit, static_argnums=(0,))
    def gaussian(self, numax, width, height):
        """ Gaussian to describe oscillations

        Note the input height is log10.

        """

        m = height * jnp.exp(-(self._f - numax)**2 / (2.0 * width**2))

        return m

    @partial(jax.jit, static_argnums=(0,))
    def model(self, params):
        """The spectrum model

        Includes 3 Harvey-like profiles, 1 Gaussian and a white noise term.

        Includes reflection around the Nyquist frequency.

        Parameters
        ----------
        params : list
            List of model parameters.
            
        Returns
        -------
        model : array
            The model evaluated at each frequency in the psd.
        _model : array
            The unreflected psd model.
        D : dict
            Dictionary of each model term.
        """
        
        theta_asy, theta_bkg, theta_extra = self.unpackParams(params)
        
        W = theta_bkg[-1]

        modes = self.asymptotic_model(theta_asy)

        bkg, comps = self.background_model(theta_bkg)
        
        _model = (1 + modes) * bkg * self.eta 
        
        model = _model[:len(self.f)] + W + _model[len(self.f):][::-1] 

        comps['modes'] = modes

        comps['W'] = W

        return model, _model, comps, theta_extra 
    
    @partial(jax.jit, static_argnums=(0,))
    def background_model(self, theta):

        hsig1, hnu1, exp1, hsig2, hnu2, exp2, hsig3, hnu3, exp3, _ = theta

        H1 = self.harvey(hsig1, hnu1, exp1)

        H2 = self.harvey(hsig2, hnu2, exp2)

        H3 = self.harvey(hsig3, hnu3, exp3)

        return H1 + H2 + H3, {'H1': H1, 'H2': H2, 'H3': H3}

    @partial(jax.jit, static_argnums=(0,))
    def unpackParams(self, theta): 
        """ Separate the model parameters into asy, bkg, reg and phi

        This will need to be updated when bkg is rolled into DR.

        Parameters
        ----------
        theta : array
            Array of parameters drawn from the posterior distribution.

        Returns
        -------
        theta_asy : array
            The unpacked parameters of the p-mode asymptotic relation.
        theta_obs : array
            The unpacked parameters of the observational parameters.
        theta_bkg : array
            The unpacked parameters of the harvey background model.
        theta_reg : array
            The unpacked parameters of the reggae model.
        phi : array
            The model scaling parameter.  

        self.labels = ['dnu', 'd02', 'alpha', 'mode_width', 'teff', 'bp_rp', 'eps',
                       'numax', 'env_height', 'env_width', 
                       'H1_power', 'H1_nu', 'H1_exp',
                       'H2_power', 'H2_nu', 'H2_exp',
                       'H3_power', 'H3_nu', 'H3_exp',
                       'white']

        self.pcalabels = ['dnu', 'd02', 'alpha', 'mode_width', 'teff', 'bp_rp', 'eps',
                              'numax', 'env_height', 'env_width',
                              'H1_power', 'H1_nu', 'H1_exp',
                              'H2_power', 'H2_nu', 'H2_exp',]

        """
        if self.with_pca:
            # The sample used for DR is in log10, so the inverse transform 
            theta_invT = self.DR.inverse_transform(theta[:self.DR.dims_R])
            
            dnu, d02, alpha, modeWidth, teff = 10**theta_invT[0:5] 
            bp_rp, eps, = theta_invT[5:7] 

            numax, envHeight, envWidth  = 10**theta_invT[7:10] 

            hsig1, hnu1 = 10**theta_invT[10:12] 
            exp1 = theta_invT[12] 
    
            hsig2, hnu2 = 10**theta_invT[13:15]
            exp2 = theta_invT[15]

            hsig3, hnu3, exp3, white = theta[self.DR.dims_R:]

            hsig3 = 10**hsig3

            w = 10**white

        else:
            numax, height, dwidth, hsig1, dhnu1, exp1, hsig2, dhnu2, exp2, hsig3, hnu3, exp3, white = theta

            height = 10**height

            # Start with Harvey laws
            hsig1 = 10**hsig1
            hnu1 = self.nuHarveyEnv(numax) * dhnu1
            
            hsig2 = 10**hsig2
            hnu2 = self.nuHarveyGran(numax) * dhnu2
        
            # Add envelope
            width = dwidth * self.envWidth(numax) * self.wfac
            
            hsig3 = 10**hsig3

            w = 10**white

        theta_asy = [numax, dnu, eps, d02, alpha, envHeight, envWidth, modeWidth]

        theta_bkg = [hsig1, hnu1, exp1, hsig2, hnu2, exp2, hsig3, hnu3, exp3, w]

        theta_extra = {'teff': teff, 
                       'bp_rp': bp_rp,
                       'hnu1': hnu1,
                       'hnu2': hnu2,
                       'hnu3': hnu3,
                       'numax': numax}

        return theta_asy, theta_bkg, theta_extra

    @partial(jax.jit, static_argnums=(0,))
    def _lnlike_std(self, mod):
        L = -jnp.sum(jnp.log(mod) + self.p/mod)
        return L

    @partial(jax.jit, static_argnums=(0,))
    def _lnlike_binned(self, mod):
        L = jnp.sum((self.psd.pbins-1) * (jnp.log(self.psd.pbins) + jnp.log(self.p)) - utils.lnfactorial(self.psd.pbins-1) - self.psd.pbins*(jnp.log(mod) + self.p/mod))
        return L

    @partial(jax.jit, static_argnums=(0,))
    def lnlike(self, params):
         
        mod, _mod, _, theta_extra = self.model(params)

        # hnu3 is bounded at 0, so hnu2 is in turn also bounded at 0 if hnu3 < hnu2. Same for hnu1 and numax.
        T = (theta_extra['hnu3'] < theta_extra['hnu2']) & \
            (theta_extra['hnu2'] < theta_extra['numax']) & \
            (theta_extra['hnu2'] < theta_extra['hnu1'])
 
        lnL = jax.lax.cond(T, self._lnlike, lambda mod: -jnp.inf, mod)
        
        lnL += self.obsLikes[0].logpdf(theta_extra['teff'])

        lnL += self.obsLikes[1].logpdf(theta_extra['bp_rp'])

        return lnL

    def runDynesty(self, nlive=300, dynamic=False, progress=False):

        tstart = time.time()
        if dynamic:
            sampler = dynesty.DynamicNestedSampler(self.lnlike, self.ptform, self.ndim, nlive=nlive, sample='rwalk')
            sampler.run_nested(print_progress=progress, wt_kwargs={'pfrac': 1.0}, dlogz_init=1e-3 * (nlive - 1) + 0.01, nlive_init=nlive)   
        else:           
            sampler = dynesty.NestedSampler(self.lnlike, self.ptform, self.ndim, nlive=nlive, sample='rwalk')
            sampler.run_nested(print_progress=progress)

        tend = time.time()
        result = sampler.results

        samples, weights = result.samples, jnp.exp(result.logwt - result.logz[-1])

        new_samples = dyfunc.resample_equal(samples, weights)

        self._samples = new_samples

        self.run_stats = {'iterations': sampler.it,
                          'ncall': sampler.ncall,
                          'eff': sampler.eff,
                          'runtime': tend-tstart}

        self.sampler = sampler

        return sampler, new_samples

    @partial(jax.jit, static_argnums=(0,))
    def sumPrior(self, params):
        
        lnPr = jnp.sum(jnp.array([self.priors[i].logpdf(params[i]) for i in range(self.ndim)]))
        
        return lnPr

    @partial(jax.jit, static_argnums=(0,))
    def lnPost(self, params):

        lnPr = self.sumPrior(params)

        lnL = self.lnlike(params)

        T = jnp.isnan(lnL) | jnp.isnan(lnPr)

        logP = jax.lax.cond(T, lambda lnPr, lnL: -jnp.inf, lambda lnPr, lnL: lnPr + lnL, lnPr, lnL)

        return logP

    def runEmcee(self, nwalkers=100, nsteps=100, burn = 0.8, accept=0.16, progress=False):

        p0 = jnp.array([self.ptform(np.random.uniform(low=0, high=1, size=self.ndim)) for i in range(nwalkers)])

        sampler = emcee.EnsembleSampler(nwalkers, self.ndim, self.lnPost)

        sampler.run_mcmc(p0, nsteps=nsteps, progress=progress);

        idx= sampler.acceptance_fraction > accept

        samples = sampler.chain[idx, int(burn*nsteps):, :]

        return sampler, samples














    def makePriorCorner(self, fig, N = 500):

        prior_samples = np.array([self.ptform(np.random.uniform(low=0, high=1, size=self.ndim)) for i in range(N)])

        self.makeSamplingCorner(fig, prior_samples)

    def makeSamplingCorner(self, fig, samples, N=500, outputDir=None):
        
        thinIdx = np.random.choice(np.arange(samples.shape[0]), size=min([N, samples.shape[0]]), replace=False)
        
        if self.with_pca:
            labels = [r'$\theta_%i$' % (i) for i in range(self.DR.dims_R)] + [x for x in self.labels if x not in self.pcalabels]
        else:
            labels = self.labels

        corner.corner(samples[thinIdx, :], labels=labels, hist_kwargs={'density': True}, fig=fig);
        
        axes = np.array(fig.axes).reshape((self.ndim, self.ndim))
        
        for i in range(self.ndim):

            utils._priorCurve(axes[i,i], self.priors[i].ppf, self.priors[i].pdf)
 
    def plotModel(self, fig, ax, samples=None, obs=None, outputDir=None, scale=1):

        def plotLines(ax, params, idx=None, alpha=1, components=True, scale=1):

            if idx is None:
                idx = np.arange(len(self.f))

            model, _model, comps, _ = self.model(params)

            ax.plot(self.f[idx], model[idx]/scale, lw =2, color = 'C3', alpha=alpha, label='Spectrum model')

            if components:

                ax.plot(self.f[idx], comps['H1'][idx], ls = 'dashed', lw = 3, color = 'C1', alpha=alpha, label='First Harvey profile')

                ax.plot(self.f[idx], comps['H2'][idx], ls = 'dashed', lw = 3, color = 'C2', alpha=alpha, label='Second Harvey profile')

                ax.plot(self.f[idx], comps['H3'][idx], ls = 'dashed', lw = 3, color = 'C5', alpha=alpha, label='Third Harvey profile')

                #ax.plot(self.f[idx], comps['modes'][idx], ls = 'dashed', lw = 3, color = 'C4', alpha=alpha)

                ax.axhline(comps['W'], color = 'k', ls='dashed', alpha=alpha, label='Shot noise level')

        if 'dnu' in obs.keys():
            fac = int(max([1, 0.05 / obs['dnu']]))
        else:
            fac = int(max([1, 0.05 / (self.f[1] - self.f[0])]))
        kernel = conv.Gaussian1DKernel(stddev=fac)
        smoo = conv.convolve(self.p, kernel)

        #ax[0].plot(self.f, self.p, 'k-', label='Data', alpha=0.2)
        ax[0].plot(self.f, smoo, 'k-', label='Smoothed', lw=2, alpha=0.2)
        ax[0].set_yscale('log')
        ax[0].set_xscale('log')
        ax[0].set_ylim(smoo.min(), smoo.max()*1.1)
        ax[0].set_xlim(self.f.min(), self.f.max())

        #ax[1].plot(self.f, self.p, 'k-', label='Data', alpha=0.2)
        ax[1].plot(self.f, smoo/scale, 'k-', label='Smoothed', lw=2, alpha=0.2)
        ax[1].set_yscale('linear')
        ax[1].set_xscale('linear')
        ax[1].set_xlim(self.obs['numax'][0] - 0.8*0.66 * self.obs['numax'][0]**0.88, 
                       self.obs['numax'][0] + 0.8*0.66 * self.obs['numax'][0]**0.88)
        idx = abs(self.f - self.obs['numax'][0]) < 0.8*0.66 * self.obs['numax'][0]**0.88
        ax[1].set_ylim([0, 1/scale*smoo[idx].max()*1.])
        ax[1].ticklabel_format(axis='y', style='sci', scilimits=(0,0))

        if samples is None:
            mu_pr = np.zeros(self.ndim) + 0.5

            params = self.ptform(mu_pr)

            plotLines(ax[0], params)
            plotLines(ax[1], params, components=False)

        else:
            nsamples = len(samples[:, 0])

            thinIdx = utils.gen_log_space(len(self.f), min([len(self.f), 5000]))

            rnd = np.random.choice(np.arange(nsamples), size=100, replace=False)

            for idx in rnd:
                plotLines(ax[0], samples[idx, :], thinIdx, alpha = 0.1)
                plotLines(ax[1], samples[idx, :], thinIdx, alpha = 0.1, components=False, scale=scale)
   
        ax[0].set_ylabel('PSD [ppm$^2/\mu$Hz]')
        
        ax[1].set_ylabel('PSD [ppm$^2/\mu$Hz]')

        if self.DR.dims_R == 16:
            ax[0].set_xlabel('Frequency [$\mu$Hz]')
            ax[1].set_xlabel('Frequency [$\mu$Hz]')
        else:
            ax[0].set_xticks([])
            ax[1].set_xticks([])

        ax[0].text(x=210, y=3e4, s=f'{self.DR.dims_R} PCs', ha='right', bbox=dict(edgecolor='k', facecolor='w', alpha=0.5))
        ax[1].text(x=45.1, y=8.225e4, s=f'{self.DR.dims_R} PCs', ha='right', bbox=dict(edgecolor='k',facecolor='w', alpha=0.5))

        ext = f'pca{self.DR.dims_R}'

        fig.tight_layout()
        if outputDir is not None:
            
            if samples is None:
                path = os.path.join(*[outputDir, self.ID + f'_prior_model_{ext}.png'])
            else:
                path = os.path.join(*[outputDir, self.ID + f'_samples_model_{ext}.png'])

            fig.savefig(path, dpi=300)

    def storeResults(self, outputDir):
            
        if self.with_pca:
            ext = f'pca{self.DR.dims_R}'
        else:
            ext = 'nopca'

        # Store the class instance
        gfitpath = os.path.join(*[outputDir, os.path.basename(outputDir) + f'_{ext}.sfit'])

        with open(gfitpath, 'wb') as outfile:
           dill.dump(self, outfile)

        # Store the packed samples
        spath = os.path.join(*[outputDir, os.path.basename(outputDir) + f'_samples_{ext}'])

        np.savez_compressed(spath, samples=self._samples)

        # Store the unpacked samples
        Nsamples = self._samples.shape[0]

        full_samples = np.zeros((Nsamples, len(self.labels)))

        for k in range(Nsamples):
            
            theta_asy, theta_bkg, theta_extra = self.unpackParams(self._samples[k, :])
            
            full_samples[k, :] = np.hstack((np.array(theta_asy), np.array(theta_bkg), np.array(theta_extra['teff']), np.array(theta_extra['bp_rp'])))

        fspath = os.path.join(*[outputDir, os.path.basename(outputDir) + f'_full_samples_{ext}'])

        np.savez_compressed(fspath, samples=full_samples)
        
    def makeFullCorner(self, fig, samples, N=500, outputDir=None):

        axes = np.array(fig.axes).reshape((len(self.labels), len(self.labels)))
        
        thinIdx = np.random.choice(np.arange(samples.shape[0]), 
                                size=min([N, samples.shape[0]]), 
                                replace=False)

        full_samples = np.zeros((N, len(self.labels)))
        
        for i, sample in enumerate(samples[thinIdx, :]):
            
            full_samples[i, :] = self.unpackParams(sample)
        
        # Convert some values back to log
        for i in list(range(10))+[12]:
            full_samples[:, i] = np.log10(full_samples[:, i])

        corner.corner(full_samples, labels=self.labels, hist_kwargs={'density': True}, fig=fig);
        
        if self.with_pca:
            Fti = self.DR.inverse_transform(self.DR.data_R)
            
            Fti_ppfs, Fti_pdfs, _, _ = self.DR.getQuantileFuncs(Fti)
            
            for i in range(self.DR.dims_F):
                
                utils._priorCurve(axes[i, i], Fti_ppfs[i], Fti_pdfs[i])

            for i, j in enumerate(range(self.DR.dims_R, self.DR.dims_R + len(self.labels) - self.DR.dims_F)):
                
                utils._priorCurve(axes[i + self.DR.dims_F, i + self.DR.dims_F], self.priors[j].ppf, self.priors[j].pdf)
 
