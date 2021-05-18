def file_length(fname):
    # Count the number of lines in a file
    # https://stackoverflow.com/questions/845058/how-to-get-line-count-cheaply-in-python
    with open(fname) as f:
        i = 0
        for i, _ in enumerate(f):
            pass
    return i+1

def array_values_at_indices(array, list_of_array_positions):
    # Returns values of the array at the list of array position integers
   if len(array.shape) == 1:
      return array[list_of_array_positions]
   elif len(array.shape) == 2:
      ix, iy = zip(*list_of_array_positions)
      result = array[ix, iy]
      return result
   else:
      ValueError('Error, this only works in either one or two dimensions at the moment')
      return None

def arange(min, max):
    # Sensible arange function
    # I hate the inbuilt numpy one with such fury that it frightens me
   from numpy import arange
   return arange(min, max+1)

def logspace(xmin, xmax, nx):
    # Return a logarithmically spaced range of numbers
   from numpy import logspace, log10
   return logspace(log10(xmin), log10(xmax), nx)

def multiply_list_elements(multiple, list):
    # Multiply all elements in a list by a constant
   return [multiple*x for x in list]

def seq_color(i, n, cmap):
    # Sequential colors
    return cmap(i/(n-1))

def colour(i):
    # Default colours in plotly
    color = 'C%d' % i
    return color

def even(num):
    # True if integer is even
    if num%2 == 0:
        return True
    else:
        return False

def odd(num):
    # True if integer is odd
    return not even(num)

def is_power_of_two(i):
    #from math import isclose
    #from numpy import log2, rint
    #lg2 = log2(i)
    #if isclose(lg2, rint(lg2)):
    #    return True
    #else:
    #    return False
    return is_power_of_n(i, 2)

def is_power_of_n(i, n):
    from math import isclose
    from numpy import rint
    from mead_special_functions import logn
    lg = logn(i, n)
    if isclose(lg, rint(lg)):
        return True
    else:
        return False

def print_array_attributes(x):
    # Print useful array attributes
    print('Array attributes')
    print('ndim:', x.ndim)
    print('shape:', x.shape)
    print('size:', x.size)
    print('dtype:', x.dtype)
    print('')

def print_array_statistics(x):
    # Print useful array statistics
    print('Array statistics')
    print('sum:', x.sum())
    print('min:', x.min())
    print('max:', x.max())
    print('mean:', x.mean())
    print('std:', x.std())
    print('std (Bessel corrected):', x.std(ddof=1))
    print()

# use numpy deg2rad() or radians()
#def degrees_to_radians(theta):
#    from numpy import pi
#    return theta*pi/180.

# use numpy rad2deg or degrees()
#def radians_to_degrees(theta):
#    from numpy import pi
#    return theta*180./pi

# A sum function that returns nan if any of the values to be summed is a nan
# This should be the default behaviour of the np.sum function
#def nansum(a, **kwargs):
#    from numpy import isnan, nan, nansum
#    if isnan(a).any():
#        return nan
#    else:
#        return nansum(a, **kwargs)