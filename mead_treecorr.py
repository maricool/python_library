import numpy as np

import mead_general as mead

def _calculate_rmin_rmax(nn):
    
    # Calculate the bin limits
    nr = nn.nbins
    if (nn.bin_type == 'Log'):
        rbins = mead.logspace(nn.min_sep, nn.max_sep, nr+1)
    elif (nn.bin_type == 'Lin'):
        rbins = np.linspace(nn.min_sep, nn.max_sep, nr+1)
    else:
        raise ValueError('bin_type is not recognised')
    
    # Fill arrays of the minimum and maximum r contributing to each bin
    rmin = np.zeros((nr))
    rmax = np.zeros((nr))
    for i in range(nr):
        rmin[i] = rbins[i]
        rmax[i] = rbins[i+1]
    
    # Return the minimum and maximum limit for each bin
    return rmin, rmax

def calculateXi_sim(nn, Vsim, N1, N2=None):
    
    # Checks
    if (nn.npairs.all() != nn.weight.all()):
        raise ValueError('This does not currently work for weighted data')
    
    # Get the limits of the bin in r
    rmin, rmax = _calculate_rmin_rmax(nn)
    nr = nn.nbins
    
    # Calculate the correlation function from the pair counts
    # TODO: Should there be N-1/N in variance given that mean is estimated from data?
    nn.xi = np.zeros((nr))
    nn.varxi = np.zeros((nr))
    for i in range(nr):
        V = (4./3.)*np.pi*(rmax[i]**3-rmin[i]**3) # Volume of (possibly thick) shell
        if N2 is None:
            N12 = (N1**2)*V/Vsim # Expected number of pairs: Auto-correlation
        else: 
            N12 = N1*N2*V/Vsim # Expected number of pairs: Cross-correlation
        nn.xi[i] = -1.+nn.npairs[i]/N12 # Construct correlation function
        nn.varxi[i] = nn.npairs[i]/N12**2 # Variance in measured correlation function assuming Poisson stats for pair counts
        
    return nn.xi, nn.varxi