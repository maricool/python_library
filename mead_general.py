# https://stackoverflow.com/questions/845058/how-to-get-line-count-cheaply-in-python
def file_length(fname):

    with open(fname) as f:
        i = 0
        for i, _ in enumerate(f):
            pass
    return i + 1

# Returns values of the array at the list of array position integers
def array_values_at_indices(array, list_of_array_positions):

   if len(array.shape) == 1:
      return array[list_of_array_positions]
   elif len(array.shape) == 2:
      ix, iy = zip(*list_of_array_positions)
      result = array[ix, iy]
      return result
   else:
      ValueError('Error, this only works in either one or two dimensions at the moment')
      return None

# Sensible arange function
def arange(min, max):
   from numpy import arange
   return arange(min, max+1)

# Return a logarithmically spaced range of numbers
def logspace(xmin, xmax, nx):
   from numpy import logspace, log10
   return logspace(log10(xmin), log10(xmax), nx)

# Multiply all elements in a list by a constant
def multiply_list_elements(multiple, list):
   return [multiple*x for x in list]

# Sequential colors
def seq_color(i, n, cmap):
    return cmap(i/(n-1))

# Default colours in plotly
def colour(i):
    color = 'C%d' % i
    return color

# Cube root
def cbrt(x):
    return x**(1./3.)

# 2D trapezium rule
def trapz2d(F, x, y):
    from numpy import zeros, trapz
    Fmid = zeros((len(y)))
    for iy, _ in enumerate(y):
        Fmid[iy] = trapz(F[:, iy], x)
    return trapz(Fmid, y)

def logx_InterpolatedUnivariateSpline(x, y, **kwargs):
    from numpy import log
    from scipy.interpolate import InterpolatedUnivariateSpline as IUS
    log_IUS = IUS(log(x), y, **kwargs)
    return lambda x: log_IUS(log(x))

def logy_InterpolatedUnivariateSpline(x, y, **kwargs):
    from numpy import log, exp
    from scipy.interpolate import InterpolatedUnivariateSpline as IUS
    log_IUS = IUS(x, log(y), **kwargs)
    return lambda x: exp(log_IUS(x))

def loglog_InterpolatedUnivariateSpline(x, y, **kwargs):
    from numpy import log, exp
    from scipy.interpolate import InterpolatedUnivariateSpline as IUS
    log_IUS = IUS(log(x), log(y), **kwargs)
    return lambda x: exp(log_IUS(log(x)))

# use numpy deg2rad
#def degrees_to_radians(theta):
#    from numpy import pi
#    return theta*pi/180.

# use numpy rad2deg
#def radians_to_degrees(theta):
#    from numpy import pi
#    return theta*180./pi

# Check if a square array is symmetric
def check_symmetric(a, rtol=1e-05, atol=1e-08):
    from numpy import allclose
    return allclose(a, a.T, rtol=rtol, atol=atol)