import numpy as np

def quadratic_roots(a, b, c):
    '''
    Provides the two real solutions for x for a quadratic a*x^2 + b*x + c = 0 
    If discriminant is zero then the two solutions will be identical
    TODO: Expand for complex solutions
    '''
    des = b**2-4.*a*c
    if (des > 0.):
        root = np.sqrt(b**2-4.*a*c)
    else:
        raise ValueError('Quadratic discriminant is negative')
    f1 = (root-b)/(2.*a)
    f2 = (-root-b)/(2.*a)
    return f1, f2