# Import statements
import numpy as np
from scipy.integrate import quad

### Continuous probability distributions ###

def normalisation(f, x1, x2, *args):
    '''
    Calculate the normalisation of a probability distribution via integration
    f: f(x) probability distribution function
    x1, x2: Limits over which to normalise (could be -inf to +inf)
    *args: Arguments to be passed to function
    '''
    norm, _ = quad(lambda x: f(x, *args), x1, x2)
    return norm

def cumulative(x, f, x1, *args):
    '''
    Compute the cumulative distribution function via integration
    '''
    C, _ = quad(lambda x: f(x,*args), x1, x)
    return C
cumulative = np.vectorize(cumulative) # Vectorise to work with array 'x'

def draw_from_distribution(x, Cx):
    '''
    Draw a random number given an cumulative probability function
    x: array of x values
    C(x): array for cumulative probability
    '''
    r = np.random.uniform(Cx[0], Cx[-1])
    xi = np.interp(r, Cx, x)    
    return xi

def moment(n, f, x1, x2, *args):
    '''
    Compute the n-th moment of a continuous distribution via integration
    n: order of moment
    f: f(x)
    x1, x2: low and high limits
    *args: arguments to be passed to function
    '''
    norm = normalisation(f, x1, x2, *args)
    m, _ = quad(lambda x: (x**n)*f(x, *args)/norm, x1, x2)
    return m

def variance(f, x1, x2, *args):
    '''
    Compute the variance of a continuous distribution via integration
    '''
    m1 = moment(1, f, x1, x2, *args)
    m2 = moment(2, f, x1, x2, *args)
    return m2-m1**2

### ###

### Drawing random numbers from continuous distributions ###

def draw_from_1D(n, f, x1, x2, nx, *args):
    '''
    Draw random numbers from a continuous 1D distribution
    n: number of draws to make from f
    f: f(x) array to draw from
    x1, x2: limits on x axis
    nx: number of points to use along x axis
    '''
    x = np.linspace(x1, x2, nx)    
    C = cumulative(x, f, x1, *args)
    xi = np.zeros(n)
    for i in range(n):
        xi[i] = draw_from_distribution(x, C)
    return xi

def draw_from_2D(n, f, x1, x2, nx, y1, y2, ny):
    '''
    Draw random numbers from a 2D distribution
    n - number of draws to make from f
    f - f(x,y) to draw from
    x1, x2 - limits on x axis
    y1, y2 - limits on y axis
    nx, ny - number of points along x and y axes
    '''
    # Pixel sizes in x and y
    dx = (x2-x1)/np.real(nx)
    dy = (y2-y1)/np.real(ny)

    # Linearly spaced arrays of values corresponding to pixel centres
    x = np.linspace(x1+dx/2., x2-dx/2., nx)
    y = np.linspace(y1+dy/2., y2-dy/2., ny)

    # Make a grid of xy coordinate pairs
    xy = np.array(np.meshgrid(x, y))
    xy = xy.reshape(2, nx*ny) # Reshape the grid (2 here coresponds to 2 coordinates: x, y)
    xy = np.transpose(xy).tolist() # Convert to a long list
    
    # Make array of function values corresponding to the xy coordinates
    X, Y = np.meshgrid(x, y)
    z = f(X, Y)      # Array of function values
    z = z.flatten() # Flatten array to create a long list of function values
    z = z/sum(z)    # Force normalisation

    # Make a list of integers linking xy coordiantes to function values
    i = list(range(z.size)) 

    # Make the random choices with probabilties proportional to the function value
    # The integer chosen by this can then be matched to the xy coordiantes
    j = np.random.choice(i,n,replace=True,p=z) 
    
    # Now match the integers to the xy coordinates
    xs = []
    ys = []
    for i in range(n):
        xi, yi = xy[j[i]]
        xs.append(xi)
        ys.append(yi)

    # Random numbers for inter-pixel displacement
    dxs = np.random.uniform(-dx/2., dx/2., n) 
    dys = np.random.uniform(-dy/2., dy/2., n)

    # Apply uniform-random displacement within a pixel
    xs = xs+dxs
    ys = ys+dys
        
    return xs, ys

### ###

### Other ###

def correlation_matrix(cov):
    '''
    Calculate a correlation matrix from a covariance matrix
    '''
    shape = cov.shape
    n = shape[0]
    if n != shape[1]:
        raise TypeError('Input covariance matrix must be square')
    cor = np.empty_like(cov)
    
    for i in range(n):
        for j in range(n):
            cor[i, j] = cov[i, j]/np.sqrt(cov[i, i]*cov[j, j]) 
            
    return cor

### ###

### Integer distributions ###

def central_condition_Poisson(n, lam, pc):
    '''
    Probability mass function for a Poisson distribution affected by the central condition
    n: p(n); n is an integer
    lam: mean value for the underlying Poisson distribution (not the mean value of this distribution)
    pc: probability of hosting a central galaxy
    '''
    from scipy.stats import poisson
    p = poisson.pmf(n, lam)
    p = np.where(n == 0, pc*p+1.-pc, pc*p)
    return p

def expectation_integer_distribution(p, f, nmax, *args):
    '''
    The expectation value of some function of an integer probability distribuion
    P(n, *args): Probability distribution (mass) function
    f(n): Function over which to compute expectation value (e.g., f(n) = n would compute mean; f(n) = n^2 second moment)
    nmax: Maximum n to compute sum
    *args: to be passed to p(n, *args)
    '''
    ns = np.arange(nmax+1)
    ps = p(ns, *args)
    return np.sum(f(ns)*ps)

def moment_integer_distribution(p, pow, nmax, *args):
    '''
    Compute the moment of an integer distribution via direct summation
    p(n, *args): Probability distribution (mass) function
    pow: Order for moment (0 - normalisation; 1 - mean; 2 - second moment)
    nmax: Maximum n to compute sum
    *args: to be passed to p(n, *args)
    '''
    return expectation_integer_distribution(p, lambda n: n**pow, nmax, *args)

def sum_integer_distribution(p, nmax, *args):
    '''
    Compute the sum of probabilities for integer distribution via direct summation (should be unity)
    p(n, *args): Probability distribution (mass) function
    nmax: Maximum n to compute sum
    *args: to be passed to p(n, *args)
    '''
    return moment_integer_distribution(p, 0, nmax, *args)

def mean_integer_distribution(p, nmax, *args):
    '''
    Compute the mean value of an integer distribution via direct summation
    p(n, *args): Probability distribution (mass) function
    nmax: Maximum n to compute sum
    *args: to be passed to p(n, *args)
    '''
    return moment_integer_distribution(p, 1, nmax, *args)

def variance_integer_distribution(p, nmax, *args):
    '''
    Compute the variance of an integer distribution via direct summation
    p(n, *args): Probability distribution (mass) function
    nmax: Maximum n to compute sum
    *args: to be passed to p(n, *args)
    '''
    mom1 = moment_integer_distribution(p, 1, nmax, *args)
    mom2 = moment_integer_distribution(p, 2, nmax, *args)
    return mom2-mom1**2

### ###
