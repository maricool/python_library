# Count the number of lines in a file
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
# I hate the inbuilt numpy one with such fury that it frightens me
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

# use numpy deg2rad
#def degrees_to_radians(theta):
#    from numpy import pi
#    return theta*pi/180.

# use numpy rad2deg
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