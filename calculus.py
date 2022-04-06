# Standard imports
import numpy as np
from scipy import integrate

def integrate_quad_log(func,a,b,\
                       args=(),\
                       full_output=0,\
                       epsabs=1.49e-08,\
                       epsrel=1.49e-08,\
                       limit=50,\
                       points=None,\
                       weight=None,\
                       wvar=None,\
                       wopts=None,\
                       maxp1=50,\
                       limlst=50):
    # A routine to integrate in log space. 
    # This may actually be pretty useless... not sure. Should do speed tests
    # TODO: Surely can use *args and **kwargs here. This is ugly as fuck.
    loga=np.log(a)
    logb=np.log(b)
    ans=integrate.quad(lambda x, *args: np.exp(x)*func(np.exp(x),*args), loga, logb,\
                       args=args,\
                       full_output=full_output,\
                       epsabs=epsabs,\
                       epsrel=epsrel,\
                       limit=limit,\
                       points=points,\
                       weight=weight,\
                       wvar=wvar,\
                       wopts=wopts,\
                       maxp1=maxp1,\
                       limlst=limlst)
    return ans

def integrate_rectangular(fx, x):
    # A very simple rectangular integration that assumes equal sized bins
    # This is zeroth order - worse than trapezium rule - but consistent for some binned data
    dx = x[1]-x[0]
    return sum(fx)*dx

def trapz2d(F, x, y):
    # Two-dimensional trapezium rule
    # First integrates along x for each y, and then y
    from numpy import zeros, trapz
    Fmid = zeros((len(y)))
    for iy, _ in enumerate(y):
        Fmid[iy] = trapz(F[:, iy], x)
    return trapz(Fmid, y)