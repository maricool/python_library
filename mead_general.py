### Functions ###
# First set involve only basic Python, then numpy, then matplotlib

### This set of functions use only basic Python ###

def opposite_side(left_or_right):
    '''
    Returns the opposite of the strings: 'left' or 'right'
    '''
    if left_or_right == 'left':
        out = 'right'
    elif left_or_right == 'right':
        out = 'left'
    else:
        raise ValueError('Input should be either \'left\' or \'right\'')
    return out

def number_name(n):
    '''
    The standard name for 10^n
    e.g., 10^2 = hundred
    e.g., 10^9 = billion
    '''
    if n == int(1e1):
        return 'ten'
    elif n == int(1e2):
        return 'hundred'
    elif n == int(1e3):
        return 'thousand'
    elif n == int(1e4):
        return 'ten thousand'
    elif n == int(1e5):
        return 'hundred thousand'
    elif n == int(1e6):
        return 'million'
    elif n == int(1e9):
        return 'billion'
    else:
        raise ValueError('Integer does not appear to have a name')

def file_length(fname):
    '''
    Count the number of lines in a file
    https://stackoverflow.com/questions/845058/how-to-get-line-count-cheaply-in-python
    '''
    with open(fname) as f:
        i = 0
        for i, _ in enumerate(f):
            pass
    return i+1

def mrange(a, b=None, step=None):
    '''
    Mead range is more sensible than range, goes (1, 2, ..., n)
    e.g., list(range(4)) = [1, 2, 3, 4]
    e.g., list(range(2, 4)) = [2, 3, 4]
    I hate the inbuilt Python 'range' with such fury that it frightens me
    '''
    if step is None:
        if b is None:
            return range(a+1)
        else:
            return range(a, b+1)
    else:
        if b is None:
            raise ValueError('If a step is specified then you must also specify start (a) and stop (b)')
        else:
            return range(a, b+1, step)

def is_float_close_to_integer(x):
    '''
    Checks if float is close to an integer value
    '''
    from math import isclose
    return isclose(x, int(x))

def bin_edges_for_integers(a, b=None):
    '''
    Defines a set of bin edges for histograms of integer data
    e.g., bin_edges_for_integers(1, 5) = 0.5, 1.5, 2.5, 3.5, 4.5, 5.5 so that integers are at centres
    e.g., bin_edges_for_integers(a) assumes a is array and goes from min-0.5 to max+0.5 in steps of unity
    '''
    if b is None:
        bin_edges = list(mrange(min(a), max(a)+1))
    else:
        bin_edges = list(mrange(a, b+1))
    bin_edges = [bin_edge-0.5 for bin_edge in bin_edges] # Centre on integers
    return bin_edges


### ###

### These functions operate on collections (lists, tuples, sets, dictionaries) ###

def create_unique_list(list_with_duplicates):
    '''
    Takes a list that may contain duplicates and returns a new list with the duplicates removed
    TODO: Check that the ordering is preserved. Is the first occurance kept and later ones discarded?
    '''
    return list(dict.fromkeys(list_with_duplicates))

def remove_list_from_list(removal_list, original_list):
    '''
    Remove items in 'removal_list' if they occur in 'original_list'
    '''
    for item in removal_list:
        if item in original_list:
            original_list.remove(item)

def second_largest(numbers):
    '''
    Returns the second-largest entry in collection of numbers
    '''
    count = 0
    m1 = m2 = float('-inf')
    for x in numbers:
        count += 1
        if x > m2:
            if x >= m1:
                m1, m2 = x, m1
            else:
                m2 = x
    return m2 if count >= 2 else None

### ###

### This set of functions use numpy ###

def arange(min, max):
    '''
    Sensible arange function
    I hate the inbuilt numpy one with such fury that it frightens me
    '''
    from numpy import arange
    return arange(min, max+1)

def logspace(xmin, xmax, nx):
    '''
    Return a logarithmically spaced range of numbers
    Numpy version is specifically base10, which is insane since log spacing is independent of base
    '''
    from numpy import logspace, log10
    return logspace(log10(xmin), log10(xmax), nx)

def is_power_of_n(x, n):
    '''
    Checks if x is a perfect power of n (e.g., 32 = 2^5)
    '''
    from mead_special_functions import logn
    lg = logn(x, n)
    return is_float_close_to_integer(lg)

def is_power_of_two(x):
    '''
    True if arguement is a perfect power of two
    '''
    return is_power_of_n(x, 2)

def is_perfect_square(x):
    '''
    Checks if argument is a perfect square (e.g., 16 = 4^2)
    '''
    from numpy import sqrt
    root = sqrt(x)
    return is_float_close_to_integer(root)

def is_perfect_triangle(x):
    '''
    Checks if argument is a perfect triangle number (e.g., 1, 3, 6, 10, ...)
    '''
    from numpy import sqrt
    n = 0.5*(sqrt(1.+8.*x)-1.)
    return is_float_close_to_integer(n)

def print_array_attributes(x):
    '''
    Print useful array attributes
    '''
    print('Array attributes')
    print('ndim:', x.ndim)
    print('shape:', x.shape)
    print('size:', x.size)
    print('dtype:', x.dtype)
    print('')

def print_full_array(xs, title=None):
    '''
    Print full array(-like) to screen with indices
    '''
    if title is not None:
        print(title)
    for ix, x in enumerate(xs):
        print(ix, x)

def array_of_nans(shape, **kwargs):
    '''
    Initialise an array of nans
    '''
    from numpy import empty, nan
    return nan*empty(shape, **kwargs)

def remove_nans_from_array(x):
    '''
    Remove nans from array x
    '''
    from numpy import isnan
    return x[~isnan(x)]

def array_values_at_indices(array, list_of_array_positions):
    '''
    Returns values of the array at the list of array position integers
    '''
    if len(array.shape) == 1:
        return array[list_of_array_positions]
    elif len(array.shape) == 2:
        ix, iy = zip(*list_of_array_positions)
        result = array[ix, iy]
        return result
    else:
        ValueError('Error, this only works in either one or two dimensions at the moment')
        return None

def print_array_statistics(x):
    '''
    Print useful array statistics
    '''
    from numpy import sqrt
    print('Array statistics')
    n = x.size
    print('size:', n)
    print('sum:', x.sum())
    print('min:', x.min())
    print('max:', x.max())
    mean = x.mean()
    print('mean:', mean)
    std = x.std()
    var = std**2
    std_bc = std*sqrt(n/(n-1))
    var_bc = std_bc**2
    print('std:', std)
    print('std (Bessel corrected):', std_bc)
    print('variance:', var)
    print('variance (Bessel corrected):', var_bc**2)
    print('<x^2>:', mean**2+var)
    print()

def standardize_array(x):
    '''
    From: https://towardsdatascience.com/pca-with-numpy-58917c1d0391
    This function standardizes an array, its substracts mean value, 
    and then divide the standard deviation.
        x: array 
        return: standardized array
    NOTE: In sklearn the 'StandardScaler' exists to do exactly this
    '''
    from numpy import zeros, empty, append
    rows, columns = x.shape
    
    standardizedArray = zeros(shape=(rows, columns))
    tempArray = zeros(rows)
    
    for col in range(columns):
        mean = x[:, col].mean(); std = x[:, col].std()
        tempArray = empty(0)
        for element in x[:, col]:
            tempArray = append(tempArray, (element-mean)/std)
        standardizedArray[:, col] = tempArray
    return standardizedArray

def covariance_matrix(sigmas, R):
    '''
    Creates an nxn covariance matrix from a correlation matrix
    Covariance is matrix multiplication of S R S where S = diag(sigmas)
    @params
        sigmas - sigmas for the diagonal
        R - correlation matrix (nxn)
    '''
    from numpy import diag, matmul
    S = diag(sigmas)
    cov = matmul(matmul(S, R), S)
    return cov

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

### ###

### matplotlib ###

def seq_color(i, n, cmap):
    '''
    Sequential colors from i=0 to i=n-1 to select n colors from cmap
    '''
    return cmap(i/(n-1))

def colour(i):
    '''
    Default colours (C0, C1, C2, ...)
    '''
    return 'C%d'%(i)

### ###