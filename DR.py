import numpy as np
import statsmodels.api as sm
import pandas as pd
import warnings, jax
import utils
import jax.numpy as jnp
from functools import partial
import corner
jax.config.update('jax_enable_x64', True)

import sys

class PCA():

    def __init__(self, numax_guess, pcalabels, weights=None, fname='PCAsample.csv', N=1000, weight_args={}):
         
        self.pcalabels = pcalabels

        self.numax_guess = numax_guess

        self.data_F, self.dims_F, self.nsamples = self.getPCAsample(fname, N)

        self.setWeights(weights, weight_args)

        self.mu  = jnp.average(self.data_F, axis=0, weights=self.weights)

        self.var = jnp.average((self.data_F-self.mu)**2, axis=0, weights=self.weights)

        self.std = jnp.sqrt(self.var)

    def setWeights(self, w, kwargs):
        """
        Set the PCA weights. If None is given then the weights are uniform.

        Parameters
        ----------
        w : np.array
            Array of weights of length equal to number of samples.

        """
         
        if w is None:
            self.weights = jnp.ones(self.nsamples)
        elif callable(w):
            self.weights = w(self, **kwargs)
        else:
            self.weights = w

    def getPCAsample(self, fname, nsamples):
        """
        Retrieve the prior samples from a provided csv file.

        Parameters
        ----------
        fname : str
            File name where prior samples are stored.
        KDEsize : int
            Number of samples from the prior to draw around the target in
            terms of numax.

        Returns
        -------
        data: jax.DeviceArray
            Samples drawn from around the target numax.
        ndim : int
            Number of variables (dimensions) in the prior sample.
        nsamples : int
            Number of samples drawn from the prior. This is expected to be
            KDEsize, but may be less if the prior is sparse around the target.
        """

        pdata = pd.read_csv(fname)

        pdata.replace([np.inf, -np.inf], np.nan, inplace=True)

        pdata.dropna(axis=0, how="any", inplace=True)

        pdata = self.getPriorSample(pdata, nsamples)
        #pdata = self.findNearest(pdata, nsamples)

        ndim = len(self.pcalabels)

        nsamples = len(pdata)

        return jnp.array(pdata.to_numpy()), ndim, nsamples

    def getPriorSample(self, pdata, KDEsize, rstate=42):
         """ Select a subsample of the prior sample data frame

         If necessary, increases the range around starting numax, until the
         sample contains at least KDEsize targets.

         Otherwise if the number of targets in the range around the input numax
         is greater than KDEsize, KDEsize samples will be drawn from the
         distribution within that range.

         Notes
         -----
         If downsampling is necessary it is done uniformly in numax. Multiplying
         idx by a Gaussian can be done to change this to a normal distribution.
         This hasn't been tested yet though.

         Parameters
         ----------
         pdata : pandas.dataFrame
             Pandas dataframe with prior samples.
         numax : list
             The estimate of numax and uncertainty in log-scale, [numax, numax_err].
         KDEsize : int
             Number of targets to include in the KDE estimation.

         Returns
         -------
         out : pandas.dataFrame
             Array of size Ndim by KDEsize to be used to compute the KDE.
         """

         idx = np.zeros(len(pdata), dtype=bool)

         numax = np.array(self.numax_guess)

         # Iteratively expand the search range around input numax until either sigma
         # limit is exceeded, or target number is reached.
         nsigma = 1

         while len(pdata[idx]) < KDEsize:

             idx = np.abs(pdata.bkg_numax.values - numax[0]) < nsigma * numax[1]

             if (np.round(nsigma, 1) < 9.0) and (len(pdata[idx]) < KDEsize):
                 nsigma += 0.1
             else:
                 break

         ntgts = len(idx[idx==1])

         # Print some warning messages if necessary
         nu_range = (10**(numax[0] + jnp.array([-nsigma, nsigma]) * numax[1])).astype(int)

         if ntgts <= 1:
             raise ValueError(f'Insufficient samples found within {jnp.round(nsigma, 1)} sigma of the provided numax. Either your errors are too small, or there are not enough samples in this range.')

         elif ntgts < KDEsize:
             warnings.warn(f'Search range {nu_range[0]} - {nu_range[1]} muHz returned {ntgts} targets. The target number was {KDEsize}.')
             KDEsize = ntgts

         # Use pandas dataframe built-in random sampler
         out = pdata.sample(KDEsize, weights=idx, replace=False, random_state=rstate).reset_index(drop=True)

         return out[self.pcalabels]


    def findNearest(self, pdata, N=100):

        numax = np.array(self.numax_guess)

        dnumax = pdata['bkg_numax'].values - numax[0] 

        sortidx = np.argsort(abs(dnumax))
        
        out = pdata.loc[sortidx, :][:N]
        
        return out[self.pcalabels]

    @partial(jax.jit, static_argnums=(0,))
    def scale(self, data):
        """
        Scale a sample of data such that it has zero mean and unit standard
        deviation.

        Parameters
        ----------
        data : jax.DeviceArray
            Sample of data

        Returns
        -------
        scaled : jax.DeviceArray
            The sample of data scaled.
        """

        scaled = (data - self.mu) / self.std

        return scaled

    @partial(jax.jit, static_argnums=(0,))
    def invert_scale(self, scaledData):
        """
        Invert the scaling of the data.

        Parameters
        ----------
        scaledData : jax.DeviceArray
            Scaled sample of data.

        Returns
        -------
        unscaled : jax.DeviceArray
            The sample of data unscaled.

        """

        unscaled = scaledData * self.std + self.mu

        return unscaled

    @partial(jax.jit, static_argnums=(0,))
    def transform(self, X):

        _X = self.scale(X)
         
        Y = self.eigvectors[:, self.sortidx].T.dot(_X.T)

        return Y.T.real

    @partial(jax.jit, static_argnums=(0,))
    def inverse_transform(self, Y):

        _X = jnp.dot(Y, self.eigvectors[:, self.sortidx].T)

        return self.invert_scale(_X).real

    def fit_weightedPCA(self, dim):
       
        self.dims_R = dim

        _X = self.scale(self.data_F)
         
        W = jnp.diag(self.weights)
        
        C = _X.T@W@_X * jnp.sum(self.weights) / (jnp.sum(self.weights)**2 - jnp.sum(self.weights**2))

        self.eigvals, self.eigvectors = jnp.linalg.eig(C)

        self.sortidx = sorted(range(len(self.eigvals)), key=lambda i: self.eigvals[i], reverse=True)[:self.dims_R]

        self.explained_variance_ratio = sorted(self.eigvals / jnp.sum(self.eigvals), reverse=True)

        self.erank = jnp.exp(-jnp.sum(self.explained_variance_ratio * np.log(self.explained_variance_ratio))).real

        self.data_R = self.transform(self.data_F)
         

    def getQuantileFuncs(self, data):
        """

        Estimate quantile function for a distribution of points, based on the
        univariate KDE of the points.

        """

        ppfs = []

        pdfs = []

        cdfs = []

        logpdfs = []

        for i in range(data.shape[1]):

            kde = sm.nonparametric.KDEUnivariate(np.array(data[:, i]).real)

            kde.fit(cut=4)

            A = jnp.linspace(0, 1, len(kde.cdf))

            cdfs.append(kde.cdf)
            
            # The icdf from statsmodels is only evaluated on the input values,
            # not the complete support of the pdf which may be wider. 
            Q = utils.getCurvePercentiles(kde.support, 
                                          kde.evaluate(kde.support),
                                          percentiles=A)

            ppfs.append(utils.jaxInterp1D(A, Q))
            
            pdfs.append(utils.jaxInterp1D(kde.support, kde.evaluate(kde.support)))

            logpdfs.append(utils.jaxInterp1D(kde.support, jnp.log(kde.evaluate(kde.support))))

        return ppfs, pdfs, logpdfs, cdfs

    def makeDRTrainingCorner(self, ):
    
        labels = [r'$\theta_%i$' % (i) for i in range(self.dims_R)]

        if not hasattr(self, 'data_R'):
            raise AttributeError('Unable to plot corner plot. Run fit_weightedPCA method first')

        fig = corner.corner(self.data_R, hist_kwargs={'density': True}, labels=labels);
        
        axes = np.array(fig.axes).reshape((self.dims_R, self.dims_R))
        
        if hasattr(self, 'ppf'):
            for i in range(self.dims_R):

                utils._priorCurve(axes[i,i], self.ppf[i], self.pdf[i])

        xlim, ylim = axes[0,0].get_xlim(), axes[0,0].get_ylim()
        
        axes[0, 0].plot([xlim[0]-1, xlim[0]-1], [ylim[0]-1, ylim[0]-1], color='k', lw=1, label='Dim. red. training sample')
        
        axes[0, 0].legend(bbox_to_anchor=(4.5, 1.05))

    def makeTrainingCorner(self, ):

        labels = self.pcalabels

        fig = corner.corner(self.data_F, hist_kwargs={'density': True}, labels=labels, color = 'k');

        if hasattr(self, 'data_R'):
            Fti = self.inverse_transform(self.data_R)
            corner.corner(Fti, hist_kwargs={'density': True}, fig=fig, labels=labels, color = 'C3')
        
        axes = np.array(fig.axes).reshape((self.dims_F, self.dims_F))

        data_F_ppfs, data_F_pdfs, _, _ = self.getQuantileFuncs(self.data_F)
        
        for i in range(self.dims_F):

            utils._priorCurve(axes[i,i], data_F_ppfs[i], data_F_pdfs[i])
 
        xlim, ylim = axes[0,0].get_xlim(), axes[0,0].get_ylim()

        axes[0, 0].plot([xlim[0]-1, xlim[0]-1], [ylim[0]-1, ylim[0]-1], color='k', lw=1, label='Original training set')
        
        axes[0, 0].plot([xlim[0]-1, xlim[0]-1], [ylim[0]-1, ylim[0]-1], color='C3', lw=1, label='Processed training set')

        axes[0, 0].legend(bbox_to_anchor=(4.5, 1.05))


 