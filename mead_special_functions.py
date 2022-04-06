import numpy as np

# Parameters
xmin_Tk = 1e-10

def Tophat_k(x):
    '''
    Normalised tophat Fourier transform function such that T(x=0)=1
    '''
    xmin = xmin_Tk
    return np.where(np.abs(x)<xmin, 1.-x**2/10., (3./x**3)*(np.sin(x)-x*np.cos(x)))

def Gaussian(x, mu, sig):
    '''
    Gaussian function such that G(x=0)=1
    NOTE: The integral over this will not be unity
    @params
        mu - mean
        sig - standard deviation
    '''
    return np.exp(-(x-mu)**2/(2.*sig**2))

def Gaussian_distribution(x, mu, sig):
    '''
    Normalised Gaussian function that integrates to zero
    NOTE: Try scipy.stats.norm
    NOTE: G(x=0) /= 0
    @params
        mu - mean
        sig - standard deviation
    '''
    A = 1./(sig*np.sqrt(2.*np.pi))
    return A*Gaussian(x, mu, sig)

def Poisson_distribution(n, mu):
    '''
    Normalised Poisson distribution
    NOTE: Try scipy.stats.poisson
    @ params
        mu - mean
    '''
    from scipy.special import gamma
    return (mu**n)*np.exp(-mu)/gamma(n+1)

def KroneckerDelta(x1, x2):
    '''
    Kronecker delta function
    '''
    return np.where(x1==x2, 1, 0)

def logn(x, n):
    '''
    Logarithm to the base n
    '''
    return np.log(x)/np.log(n)