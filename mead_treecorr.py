import numpy as np

import treecorr

import mead_general as mead

def calculateXi_sim(nn, Vsim, N1, N2=None):
    
    r"""Calculate the correlation function from the pair counts assuming that
    the mean background density is entirely uniform (i.e., in the case of a simulation)
    
    Parameters:
        nn (NNCorrelation): The NN correlation of the number data (either auto or cross correlation)
        Vsim (float): Volume of simulation in units^3 for whatever the units of r is in the correlation
        N1 (int): Number of objects in catalog 1
        N2 (int): Number of objects in catalog 2 (optional; cross correlation only)

    """
    
    # Checks
    if nn.npairs.all() != nn.weight.all():
        raise ValueError('This does not currently work for weighted data')
    if nn.metric != 'Periodic':
        raise ValueError('This only works for periodic data')
    
    # Get the limits of the bin in r
    rmin, rmax = nn.left_edges, nn.right_edges
    nr = nn.nbins
    
    # Calculate the correlation function from the pair counts
    # TODO: Consider the case of 2D data (volume of shell -> area of annulus)
    # TODO: Should there be N/(N-1) in variance given that mean is estimated from data?
    nn.xi = np.zeros((nr))
    nn.varxi = np.zeros((nr))
    for i in range(nr):
        #V = np.pi*(rmax[i]**2-rmin[i]**2) # Area of (possibly thick) annulus
        V = (4./3.)*np.pi*(rmax[i]**3-rmin[i]**3) # Volume of (possibly thick) shell
        if N2 is None:
            N12 = (N1**2)*V/Vsim # Expected number of pairs: Auto-correlation
        else: 
            N12 = N1*N2*V/Vsim # Expected number of pairs: Cross-correlation
        nn.xi[i] = -1.+nn.npairs[i]/N12 # Construct correlation function
        nn.varxi[i] = nn.npairs[i]/N12**2 # Variance in measured correlation function assuming Poisson stats for pair counts
        
    return nn.xi, nn.varxi

def get_Maps(cat, theta_min, theta_max, ntheta, bin_slop):

    config = {
        'min_sep': theta_min,
        'max_sep': theta_max,
        'nbins': ntheta,
        'bin_slop': bin_slop,
        'sep_units': 'arcmin',
        'verbose': 1,
    }

    gg = treecorr.GGCorrelation(**config)
    gg.process(cat)

    theta = gg.meanr
    Mapsq, Mapsq_im, Mxsq, Mxsq_im, varMapsq = gg.calculateMapSq()

    return theta, Mapsq, varMapsq