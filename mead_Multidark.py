import numpy as np

# Available Multidark scale factors
scalefacs = [0.257, 0.287, 0.318, 0.348, 0.378, 0.409, 0.439, 
    0.470, 0.500, 0.530, 0.561, 0.591, 0.621, 0.652, 
    0.682, 0.713, 0.728, 0.743, 0.758, 0.773, 0.788, 
    0.804, 0.819, 0.834, 0.849, 0.864, 0.880, 0.895, 
    0.910, 0.925, 0.940, 0.956, 0.971, 0.986, 1.001]

# Snapshots corresponding to scale factors
snaps = [36, 38, 40, 42, 44, 46, 48, 
            50, 52, 54, 56, 58, 60, 62, 
            64, 66, 67, 68, 69, 70, 71, 
            72, 73, 74, 75, 76, 77, 78, 
            79, 80, 81, 82, 83, 84, 85]

# Dictionary mapping snapshots to scale factors
snaps_scalefacs = dict(zip(snaps, scalefacs))

L = 1000. # Box size [Mpc/h]

def z_from_snapshot(snap):

    a = snaps_scalefacs[snap]
    z = -1.+1./a
    return z

def snapshot_from_z(z):

    if z == 0.:
        snap = 85
    elif z == 0.5:
        snap = 52
    else:
        raise ValueError('Cannot convert redshift to snapshot')
    return snap

def read_binstats(infile):

    data = np.loadtxt(infile)

    Mmin = 10**data[:, 0]
    Mmax = 10**data[:, 1]
    M = 10**data[:, 2]
    numin = data[:, 3]
    numax = data[:, 4]
    nu = data[:, 5]
    b = data[:, 6]
    rv = data[:, 7]

    return Mmin, Mmax, M, numin, numax, nu, b, rv

def read_halo_catalogue(infile, Mdef='Mvir'):

    # Read data file in to memory
    data = np.loadtxt(infile, comments='"', delimiter=',')

    # Position data
    x = data[:, 1]
    y = data[:, 2]
    z = data[:, 3]
    
    # Halo-mass data
    if Mdef == 'Mvir':
        M = data[:, 4]
    elif Mdef == 'Mvir_all' or Mdef == 'Mtot':
        M = data[:, 5]
    elif Mdef == 'M200':
        M = data[:, 6]
    elif Mdef == 'M200c':
        M = data[:, 7]
    elif Mdef == 'M500c':
        M = data[:, 8]
    elif Mdef == 'M2500c':
        M = data[:, 9]
    else:
        raise ValueError('Mdef not recognised')

    # Return the positions and the halo masses
    return [x, y, z], M

def read_Bnl(inbase, verbose=False):
        
    # Get halo masses
    #infile = inbase+'_binstats.dat'
    #print('Input file:', infile)
    #data = np.loadtxt(infile)
    #print('Full data:')
    #print(data)
    #print('')
    infile = inbase+'_binstats.dat'
    _, _, Ms, _, _, nus, _, rvs = read_binstats(infile)

    # Halo masses
    if verbose:
        print('Halo masses [Msun/h]:')
        print(Ms)
        print('')
        print('log10 halo masses [Msun/h]:')
        print(np.log10(Ms))
        print('')

    # nu
    if verbose:
        print('nus:')
        print(nus)
        print('')

    # Virial radii
    if verbose:
        print('Halo radii [Mpc/h]:')
        print(rvs)
        print('')

    # Get wavenumbers
    infile = inbase+'_bin1_bin1_power.dat'
    data = np.loadtxt(infile)
    ks = data[:, 0]
    if verbose:
        print('File:', infile)
        print('Wavenumbers')
        print(ks)
        print('Number of k points in measurement:', len(ks))
        print('')

    # Read the measured beta_NL and Pk_hh data
    beta_NL = np.zeros((len(Ms), len(Ms), len(ks)))
    beta_NL_err = np.empty_like(beta_NL)
    Pk_hh = np.empty_like(beta_NL)
    Pk_hh_err = np.empty_like(beta_NL)
    for iM1, _ in enumerate(Ms):
        for iM2, _ in enumerate(Ms):
            bin1 = 'bin'+str(iM1+1)
            bin2 = 'bin'+str(iM2+1)    
            infile = inbase+'_'+bin1+'_'+bin2+'_power.dat'
            data = np.loadtxt(infile)
            Pk_hh[iM1, iM2, :] = data[:, 1]-data[:, 2]
            Pk_hh_err[iM1, iM2, :] = data[:, 4]
            beta_NL[iM1, iM2, :] = data[:, 5]-1.
            beta_NL_err[iM1, iM2, :] = data[:, 6]

    return Ms, nus, rvs, ks, Pk_hh, Pk_hh_err, beta_NL, beta_NL_err

def write_Bnl(k, BNL, outfile):
    
    # Number of halo masses
    nM = len(BNL[:, 0, 0])

    # Loop and write results
    data = []
    for iM1 in range(nM):
        for iM2 in range(nM):
            dat = list(zip(k, 1.+BNL[iM1, iM2, :]))
            data.extend(dat)
    np.savetxt(outfile, data)