import numpy as np

# Parameters
xmin_Tk = 1e-5

def Tophat_k(x):
    # Normalised tophat Fourier transform function such that T(x=0)=1
    xmin = xmin_Tk
    return np.where(np.abs(x)<xmin, 1.-x**2/10., (3./x**3)*(np.sin(x)-x*np.cos(x)))

def Gaussian(x, mu, sig):
    # Normalised Gaussian function such that G(x=0)=1
    # NOTE: The integral over this will not be unity
    # mu - mean
    # sig - standard deviation
    return np.exp(-(x-mu)**2/(2.*sig**2))

def KroneckerDelta(x1, x2):
    # Kronecker delta function
    return np.where(x1==x2, 1, 0)

# np.cbrt exists
#def cbrt(x):
#    # Cube root
#    return x**(1./3.)

def logn(x, n):
    # Logarithm to the base n
    return np.log(x)/np.log(n)