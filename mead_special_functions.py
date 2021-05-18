import numpy as np

# Tophat Fourier transform function
# Normalised such that T(x=0)=1
def Tophat_k(x):
    xmin=1e-5
    return np.where(np.abs(x)<xmin, 1.-x**2/10., (3./x**3)*(np.sin(x)-x*np.cos(x)))

# Gaussian function
# Normalised such that G(x=0)=1
# mu - mean
# sig - standard deviation
def Gaussian(x, mu, sig):
    return np.exp(-(x-mu)**2/(2.*sig**2))

# Kronecker delta function
def KroneckerDelta(x1, x2):
    return np.where(x1==x2, 1., 0.)

# Cube root
def cbrt(x):
    return x**(1./3.)

# Logarithm to the base n
def logn(x, n):
    return np.log(x)/np.log(n)