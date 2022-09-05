#import numpy as np
import jax.numpy as jnp
import numpy as np
import jax.scipy.special as jsp
import jax
from functools import partial
import scipy.integrate as si
import scipy.special as sc
import scipy.stats as st
from matplotlib.pyplot import *

@jax.jit
def to_log10(x, xerr):
    """ Transform to value to log10

    Takes a value and related uncertainty and converts them to logscale.
    Approximate.

    Parameters
    ----------
    x : float
        Value to transform to logscale
    xerr : float
        Value uncertainty

    Returns
    -------
    logval : list
        logscaled value and uncertainty

    """



    L = jax.lax.cond(xerr > 0,
                     lambda A: jnp.array([jnp.log10(A[0]), A[1]/A[0]/jnp.log(10.0)]),
                     lambda A: A,
                     jnp.array([x, xerr]))

    return L


class jaxInterp1D():
    """ Replacement for scipy.interpolate.interp1d in jax"""

    def __init__(self, xp, fp, left=None, right=None, period=None):
        self.xp = xp
        self.fp = fp
        self.left = left
        self.right = right
        self.period = period

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, x):

        return jnp.interp(x, self.xp, self.fp, self.left, self.right, self.period)

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


class distribution():

    def __init__(self, pdf, ppf):

        self.pdf = pdf

        self.ppf = ppf


class uniform():

    def __init__(self, **kwargs):
        allowed_keys = {'loc', 'scale'}

        self.__dict__.update((k, v) for k, v in kwargs.items() if k in allowed_keys)

        self.a = self.loc
        self.b = self.loc+self.scale
        self.mu = 0.5 * (self.a + self.b)

    def pdf(self, x):


        x = jnp.array(x)

        if isinstance(x, (jnp.ndarray, )):

            idx = (self.a <= x) & (x <= self.b)

            y = jnp.zeros_like(x)

            y = y.at[idx].set(1/self.scale)

        elif isinstance(x, (float, int)):
            if (x <= self.a) or (self.b <= x):
                y = 0

            else:
                y = 1/self.scale

        return y

    def logpdf(self, x):

        x = jnp.array(x)

        if isinstance(x, (jnp.ndarray, )):

            y = jnp.zeros_like(x) - jnp.inf

            idx = (self.a <= x) & (x <= self.b)

            y = y.at[idx].set(-jnp.log(self.scale))

        elif isinstance(x, (float, int)):
            if (x <= self.a) or (self.b <= x):
                y = -jnp.inf

            else:
                y = -jnp.log(self.scale)

        return y

    def cdf(self, x):

        x = jnp.array(x)

        if isinstance(x, (jnp.ndarray, )):

            y = jnp.zeros_like(x)

            idx = (self.a <= x) & (x <= self.b)

            _y = (x[idx] - self.a) / (self.b - self.a)

            y = y.at[idx].set(_y)

            y = y.at[x>self.b].set(1)

        elif isinstance(x, (float, int)):
            if x == self.a:
                y = 0

            elif x > self.b:
                y = 1

            else:
                y = (x - self.a) / (self.b - self.a)

        return y

    def ppf(self, y):

        y = jnp.array(y)

        x = y * (self.b - self.a) + self.a

        return x



class beta():
    def __init__(self, **kwargs):
        """ beta distribution class

        Create instances a probability density which follows the beta
        distribution.

        Parameters
        ----------
        a : float
            The first shape parameter of the beta distribution.
        b : float
            The second shape parameter of the beta distribution.
        loc : float
            The lower limit of the beta distribution. The probability at this
            limit and below is 0.
        scale : float
            The width of the beta distribution. Effectively sets the upper
            bound for the distribution, which is loc+scale.
        eps : float, optional
            Small fudge factor to avoid dividing by 0 etc.
        """

        allowed_keys = {'a', 'b', 'loc', 'scale', 'eps'}

        self.__dict__.update((k, v) for k, v in kwargs.items() if k in allowed_keys)

        self.fac = jnp.exp(jsp.gammaln(self.a + self.b)) / (jnp.exp(jsp.gammaln(self.a)) * jnp.exp(jsp.gammaln(self.b))) / self.scale

        self.logfac = jnp.log(self.fac)

        self.am1 = self.a-1

        self.bm1 = self.b-1

    def _transformx(self, x):
        """ Transform x

        Translates and scales the input x to the unit interval according to
        the loc and scale parameters.

        Parameters
        ----------
        x : array
            Input support for the probability density.

        Returns
        -------
        _x : array
            x translated and scaled to the range 0 to 1.

        """
        return (x - self.loc) / self.scale

    def _inverse_transform(self, x):

        return x * self.scale + self.loc

    def pdf(self, x, norm=True):
        """ Return PDF

        Returns the beta distribution at x. The distribution is normalized to
        unit integral by default so that it may be used as a PDF.

        In some cases the normalization is not necessary, and since it's
        marginally slower it may as well be left out.

        Parameters
        ----------
        x : array
            Input support for the probability density.
        norm : bool, optional
            If true, returns the normalized beta distribution. The default is
            True.

        Returns
        -------
        y : array
            The value of the beta distribution at x.

        """

        x = jnp.array(x)

        _x = self._transformx(x)

        if isinstance(x, (jnp.ndarray, )):

            y = jnp.zeros_like(_x)

            idx = (0 < _x) & (_x < 1)

            _y = _x[idx]**self.am1 * (1 - _x[idx])**self.bm1

            y = y.at[idx].set(_y)


        elif isinstance(x, (float, int)):
            if _x <=0 or 1 <= _x:
                y = -jnp.inf

            else:
                y = _x**self.am1 * (1 - _x)**self.bm1

        if norm:
            return y * self.fac

        else:
            return y

    def logpdf(self, x, norm=True):
        """ Return log-PDF

        Returns the log of the beta distribution at x. The distribution is
        normalized to unit integral (in linear units) by default so that it
        may be used as a PDF.

        In some cases the normalization is not necessary, and since it's
        marginally slower it may as well be left out.

        Parameters
        ----------
        x : array
            Input support for the probability density.
        norm : bool, optional
            If true, returns the normalized beta distribution. The default is
            True.

        Returns
        -------
        y : array
            The value of the logarithm of the beta distribution at x.
        """

        x = jnp.array(x)

        _x = self._transformx(x)

        if isinstance(x, (jnp.ndarray)):

            y = jnp.zeros_like(_x) - jnp.inf

            idx = (0 < _x) & (_x < 1)

            _y = self.am1 * jnp.log(_x[idx]) + self.bm1 * jnp.log(1-_x[idx])

            y = y.at[idx].set(_y)

        elif isinstance(x, (float, int)):
            if _x <=0 or _x>=1:

                y = -jnp.inf

            else:
                y = self.am1 * jnp.log(_x) + self.bm1 * jnp.log(1-_x)

        if norm:
            return y + self.logfac
        else:
            return y

    def cdf(self, x):

        _x = self._transformx(x)

        y = jsp.betainc(self.a, self.b, _x)

        y = y.at[_x<=0].set(0)

        y = y.at[_x>=1].set(1)

        return y

    def ppf(self, y):

        _x = betaincinv(self.a, self.b, y)

        x = self._inverse_transform(_x)

        return x

class normal():

    def __init__(self, **kwargs):
        """ normal distribution class

        Create instances a probability density which follows the normal
        distribution.

        Parameters
        ----------

        mu : float
            The mean of the normal distribution.
        sigma : float
            The standard deviation of the normal distribution.
        """

        allowed_keys = {'mu', 'sigma'}

        self.__dict__.update((k, v) for k, v in kwargs.items() if k in allowed_keys)

        self.norm = 1 / (jnp.sqrt(2*jnp.pi) * self.sigma)

        self.lognorm = jnp.log(self.norm)

        self.fac = -0.5 / self.sigma**2

    def pdf(self, x, norm=True):
        """ Return PDF

        Returns the normal distribution at x. The distribution is normalized to
        unit integral by default so that it may be used as a PDF.

        In some cases the normalization is not necessary, and since it's
        marginally slower it may as well be left out.

        Parameters
        ----------
        x : array
            Input support for the probability density.
        norm : bool, optional
            If true, returns the normalized normal distribution. The default is
            True.

        Returns
        -------
        y : array
            The value of the normal distribution at x.

        """
        y = jnp.exp( self.fac * (x - self.mu)**2)

        if norm:
            return y * self.norm
        else:
            return y

    def logpdf(self, x, norm=True):
        """ Return log-PDF

        Returns the log of the normal distribution at x. The distribution is
        normalized to unit integral (in linear units) by default so that it
        may be used as a PDF.

        In some cases the normalization is not necessary, and since it's
        marginally slower it may as well be left out.

        Parameters
        ----------
        x : array
            Input support for the probability density.
        norm : bool, optional
            If true, returns the normalized normal distribution. The default is
            True.

        Returns
        -------
        y : array
            The value of the logarithm of the normal distribution at x.
        """

        y = self.fac * (x - self.mu)**2

        if norm:
            return y + self.lognorm

        else:
            return y

    def cdf(self, x):

        y = 0.5 * (1 + jsp.erf((x-self.mu)/(jnp.sqrt(2)*self.sigma)))

        return y

    def ppf(self, y):

        x = self.mu + self.sigma*jnp.sqrt(2)*jsp.erfinv(2*y-1)

        return x


#from jax.scipy.special import betaln, betainc
#from jax import jit

@jax.jit
def update_x(x, a, b, p, a1, b1, afac):
    err = jsp.betainc(a, b, x) - p
    t = jnp.exp(a1 * jnp.log(x) + b1 * jnp.log(1.0 - x) + afac)
    u = err/t
    tmp = u * (a1 / x - b1 / (1.0 - x))
    t = u/(1.0 - 0.5 * jnp.clip(tmp, a_max=1.0))
    x -= t
    x = jnp.where(x <= 0., 0.5 * (x + t), x)
    x = jnp.where(x >= 1., 0.5 * (x + t + 1.), x)

    return x, t

@jax.jit
def func_1(a, b, p):
    pp = jnp.where(p < .5, p, 1. - p)
    t = jnp.sqrt(-2. * jnp.log(pp))
    x = (2.30753 + t * 0.27061) / (1.0 + t * (0.99229 + t * 0.04481)) - t
    x = jnp.where(p < .5, -x, x)
    al = (jnp.power(x, 2) - 3.0) / 6.0
    h = 2.0 / (1.0 / (2.0 * a - 1.0) + 1.0 / (2.0 * b - 1.0))
    w = (x * jnp.sqrt(al + h) / h)-(1.0 / (2.0 * b - 1) - 1.0/(2.0 * a - 1.0)) * (al + 5.0 / 6.0 - 2.0 / (3.0 * h))
    return a / (a + b * jnp.exp(2.0 * w))

@jax.jit
def func_2(a, b, p):
    lna = jnp.log(a / (a + b))
    lnb = jnp.log(b / (a + b))
    t = jnp.exp(a * lna) / a
    u = jnp.exp(b * lnb) / b
    w = t + u

    return jnp.where(p < t/w, jnp.power(a * w * p, 1.0 / a), 1. - jnp.power(b *w * (1.0 - p), 1.0/b))

@jax.jit
def compute_x(p, a, b):
    return jnp.where(jnp.logical_and(a >= 1.0, b >= 1.0), func_1(a, b, p), func_2(a, b, p))

@jax.jit
def betaincinv(a, b, p):
    a1 = a - 1.0
    b1 = b - 1.0

    ERROR = 1e-8

    p = jnp.clip(p, a_min=0., a_max=1.)

    x = jnp.where(jnp.logical_or(p <= 0.0, p >= 1.), p, compute_x(p, a, b))

    afac = - jsp.betaln(a, b)
    stop  = jnp.logical_or(x == 0.0, x == 1.0)
    for i in range(10):
        x_new, t = update_x(x, a, b, p, a1, b1, afac)
        x = jnp.where(stop, x, x_new)
        stop = jnp.where(jnp.logical_or(jnp.abs(t) < ERROR * x, stop), True, False)

    return x


def _priorCurve(ax, ppf, pdf, split=True):
    
    if split:
        xl = jnp.linspace(ppf(0.001), ppf(0.5), 101)

        xu = jnp.linspace(ppf(0.5), ppf(0.999), 101)

        ax.plot(xl, pdf(xl), color='C0', lw=4, label='PDF below median', alpha=0.5)

        ax.plot(xu, pdf(xu), color='C1', lw=4, label='PDF above median', alpha=0.5)
    else:
        x = jnp.linspace(ppf(0.001), ppf(0.999), 101)

        ax.plot(x, pdf(x), color='C0', lw=4, label='KDE of transformed prior sample', ls='dashed', alpha=0.5)

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

def getCurvePercentiles(x, y, cdf = None, percentiles=None):
    """ Compute percentiles of value along a curve

    Computes the cumulative sum of y, normalized to unit maximum. The returned
    percentiles values are where the cumulative sum exceeds the requested
    percentiles.

    Parameters
    ----------
    x : array
        Support for y.
    y : array
        Array
    percentiles: array

    Returns
    -------
    percs : array
        Values of y at the requested percentiles.
    """
    if percentiles is None:
        percentiles = [0.5 - sc.erf(n/np.sqrt(2))/2 for n in range(-2, 3)][::-1]

    y /= np.trapz(y, x)

    if cdf is None:
        cdf = si.cumtrapz(y, x, initial=0)
        cdf /= cdf.max()  
    
    percs = np.zeros(len(percentiles))

    for i, p in enumerate(percentiles):
        
        q = x[cdf >= p]

        # if len(q) == 0:
        #     percs[i] = percs[i-1]
        # else:
        
        percs[i] = q[0]

    return np.sort(percs)

def plotScatter(samples, keys, fig=None, c=None, indx=None):
    
    if indx is None:
        indx = np.zeros(samples.shape[0], dtype=bool)
    
    N = len(keys)
    
    if fig is None:
        fig, ax = subplots(N, N , figsize=(20,20))


    for i in range(N):
     
        for j in range(N):

            xdata = samples[keys[j]]
            if 'exp' in keys[j]:
                xdata = 10**xdata

            ydata = samples[keys[i]]
            if 'exp' in keys[i]:
                ydata = 10**ydata

            xlims = np.nanpercentile(xdata, [0.001, 99.999])
             
            ylims = np.nanpercentile(ydata, [0.001, 99.999])

            if i == j:

                
                y_in = ydata[~indx]

                K = st.gaussian_kde(y_in[~np.isnan(y_in)], bw_method=0.15)

                _x = np.linspace(xlims[0], xlims[1], 200)

                ax[i, j].plot(_x, K(_x), color='C0')

                ax[i, j].fill_between(_x, K(_x), color='C0', alpha=1)
                
                y_out = ydata[indx]
                 
                if len(y_out) > 0:
                    K = st.gaussian_kde(y_out[~np.isnan(y_out)], bw_method=0.15)

                    _x = np.linspace(xlims[0], xlims[1], 200)

                    ax[i, j].plot(_x, K(_x), color='C1')

                    ax[i, j].fill_between(_x, K(_x), color='C1', alpha=0.5)
    
                ax[i, j].set_yticks([])

                ax[i, j].set_xlim(xlims[0], xlims[1])

                #ax[i, j].set_ylim(0, max(K(_x))*1.1)

            elif i > j:

                ax[i, j].scatter(xdata[~indx], ydata[~indx], s=10, alpha=0.5, c=c)

                # Plot outliers if any
                if indx is not None:
                    ax[i, j].scatter(xdata[indx], ydata[indx], s=20, alpha=1, color='C1')

                ax[i, j].set_xlim(xlims[0], xlims[1])

                ax[i, j].set_ylim(ylims[0], ylims[1])

            else:
                ax[i, j].axis('off')


            if i < N-1:
                ax[i, j].set_xticks([])

            if (j > 0) & (j < N-1):
                ax[i, j].set_yticks([])


            if (i == N-1):
                ax[i, j].set_xlabel(keys[j])

            if (i > 0) and (j == 0):
                    ax[i, j].set_ylabel(keys[i])
            
    fig.tight_layout()

    return fig, ax


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