# Standard imports
import numpy as np

# Other imports
from dark_emulator import darkemu

# My imports
import mead_constants as const
import mead_general as mead
import mead_cosmology as cosmo

# Constants
dc = 1.686      # Collapse threshold for nu definition
Dv = 200.       # Spherical-overdensity halo definition
np_min = 200    # Minimum number of halo particles
npart = 2048    # Cube root of number of simulation particles
Lbox_HR = 1000. # Box size for high-resolution simulations [Mpc/h]
Lbox_LR = 2000. # Box size for low-resolution simulations [Mpc/h]

# Maximum redshift
zmax = 1.48

# Minimum and maximum values of cosmological parameters in the emulator
wb_min = 0.0211375
wb_max = 0.0233625
wc_min = 0.10782
wc_max = 0.13178
Om_w_min = 0.54752
Om_w_max = 0.82128
lnAs_min = 2.4752
lnAs_max = 3.7128
ns_min = 0.916275
ns_max = 1.012725
w_min = -1.2
w_max = -0.8

# Fiducial cosmology
wb_fid = 0.02225
wc_fid = 0.1198
Om_w_fid = 0.6844
lnAs_fid = 3.094
ns_fid = 0.9645
w_fid = -1.

# Parameters
log_interp_sigma = True

class cosmology():

    def __init__(self, wb=wb_fid, wc=wc_fid, Om_w=Om_w_fid, lnAs=lnAs_fid, ns=ns_fid, w=w_fid):

        # Primary parameters
        self.wb = wb
        self.wc = wc
        self.Om_w = Om_w
        self.lnAs = lnAs
        self.ns = ns
        self.w = w

        # Fixed parameters
        self.wnu = 0.00064
        self.Om_k = 0.

        # Derived parameters
        self.wm = self.wc+self.wb+self.wnu
        self.Om_m = 1.-self.Om_w # Flatness
        self.h = np.sqrt(self.wm/self.Om_m)
        self.Om_b = self.wb/self.h**2
        self.Om_c = self.wc/self.h**2
        self.As = np.exp(self.lnAs)/1e10
        self.m_nu = self.wnu*const.nuconst
        self.Om_nu = self.wnu/self.h**2

    def print(self):

        # Write primary parameters to screen
        print('Dark Quest primary parameters')
        print('omega_b: %1.4f' % (self.wb))
        print('omega_c: %1.4f' % (self.wc))  
        print('Omega_w: %1.4f' % (self.Om_w))
        print('As [1e9]: %1.4f' % (self.As*1e9))
        print('ns: %1.4f' % (self.ns))
        print('w: %1.4f' % (self.w))
        print()

        #print('Dark Quest fixed parameters')
        #print('omega_nu: %1.4f' % (self.wnu))
        #print('Omega_k: %1.4f' % (self.Om_k))
        #print()

        # Write derived parameters to screen
        print('Dark Quest derived parameters')
        print('Omega_m: %1.4f' % (self.Om_m))
        print('Omega_b: %1.4f' % (self.Om_b))      
        print('omega_m: %1.4f' % (self.wm))
        print('h: %1.4f' % (self.h))      
        print('Omega_c: %1.4f' % (self.Om_c))
        print('Omega_nu: %1.4f' % (self.Om_nu))      
        print('m_nu [eV]: %1.4f' % (self.m_nu))
        print()

# Random cosmological parameters from the Dark Quest hypercube
def random_cosmology():

    wb = np.random.uniform(wb_min, wb_max)
    wc = np.random.uniform(wc_min, wc_max)
    Om_w = np.random.uniform(Om_w_min, Om_w_max)
    lnAs = np.random.uniform(lnAs_min, lnAs_max)
    ns = np.random.uniform(ns_min, ns_max)
    w = np.random.uniform(w_min, w_max)

    cpar = cosmology(wb=wb, wc=wc, Om_w=Om_w, lnAs=lnAs, ns=ns, w=w)
    return cpar

def named_cosmology(name):

    # Parameters
    low_fac = 0.15
    high_fac = 0.85

    # Start from fiducial cosmology
    wb = wb_fid
    wc = wc_fid
    Om_w = Om_w_fid
    lnAs = lnAs_fid
    ns = ns_fid
    w = w_fid 

    # Vary
    if name in ['low w_b', 'low w_c', 'low Om_w', 'low lnAs', 'low ns', 'low w', 
        'high w_b', 'high w_c', 'high Om_w', 'high lnAs', 'high ns', 'high w']:
        if name in ['low w_b', 'low w_c', 'low Om_w', 'low lnAs', 'low ns', 'low w']:
            fac = low_fac
        elif name in ['high w_b', 'high w_c', 'high Om_w', 'high lnAs', 'high ns', 'high w']:
            fac = high_fac
        else:
            raise ValueError('Cosmology name not recognised')
        if name in ['low w_b', 'high w_b']:
            wb = wb_min+(wb_max-wb_min)*fac
        elif name in ['low w_c', 'high w_c']:
            wc = wc_min+(wc_max-wc_min)*fac
        elif name in ['low Om_w', 'high Om_w']:
            Om_w = Om_w_min+(Om_w_max-Om_w_min)*fac
        elif name in ['low lnAs', 'high lnAs']:
            lnAs = lnAs_min+(lnAs_max-lnAs_min)*fac
        elif name in ['low ns', 'high ns']:
            ns = ns_min+(ns_max-ns_min)*fac
        elif name in ['low w', 'high w']:
            w = w_min+(w_max-w_min)*fac
        else:
            raise ValueError('Cosmology name not recognised')
    elif name == 'Multidark':
            wb = 0.0230
            wc = 0.1093
            Om_w = 0.73
            lnAs = 3.195
            ns = 0.95
            w = -1.
    else:
        raise ValueError('Cosmology name not recognised')

    cpar = cosmology(wb=wb, wc=wc, Om_w=Om_w, lnAs=lnAs, ns=ns, w=w)
    return cpar

# Create a set of my cosmological parameters from a Dark Quest set
def create_mead_cosmology(cpar, verbose=False):

    # Make a mead cosmology
    cosm = cosmo.cosmology(Om_m=cpar.Om_m, Om_b=cpar.Om_b, Om_w=cpar.Om_w, h=cpar.h, 
                           As=cpar.As, ns=cpar.ns, w=cpar.w, m_nu=cpar.m_nu)

    # Print to screen
    if verbose:
        cosm.print()

    return cosm

# Convert my cosmology into a Dark Quest cosmology
def convert_mead_cosmology(cosm):

    # Get Dark Quest parameters from my structure
    wb = cosm.w_b
    wc = cosm.w_c
    Om_w = cosm.Om_w
    lnAs = np.log(cosm.As*1e10)
    ns = cosm.ns
    w = cosm.w
    cpar = cosmology(wb=wb, wc=wc, Om_w=Om_w, lnAs=lnAs, ns=ns, w=w)

    return  cpar

# Initialise the emulator for a given set of cosmological parameters
def init_emulator(cpar):

    # Start Dark Quest
    print('Initialize Dark Quest')
    emu = darkemu.base_class()
    print('')

    # Initialise emulator
    cparam = np.array([cpar.wb, cpar.wc, cpar.Om_w, cpar.lnAs, cpar.ns, cpar.w])   
    cpar.print()
    emu.set_cosmology(cparam) # I think that this does a load of emulator init steps

    return emu

# Matter power spectrum
def Pk_mm(emu, ks, zs, nonlinear=False):

    if isinstance(zs, float):
        if nonlinear:
            Pk = emu.get_pknl(ks, zs)
        else:         
            Pk = emu.get_pklin_from_z(ks, zs)
    else:
        Pk = np.zeros((len(zs), len(ks)))
        for iz, z in enumerate(zs):
            if nonlinear:
                Pk[iz, :] = emu.get_pknl(ks, z)
            else:         
                Pk[iz, :] = emu.get_pklin_from_z(ks, z)

    return Pk

def minimum_halo_mass(emu):

    # Minimum halo mass for the set of cosmological parameters

    Mbox_HR = comoving_matter_density(emu)*Lbox_HR**3
    mmin = Mbox_HR*np_min/npart**3
    return mmin

    # Comoving matter density
def comoving_matter_density(emu):

    Om_m = emu.cosmo.get_Omega0()
    #rhom = const.rhoc*Om_m
    rhom = cosmo.comoving_matter_density(Om_m)
    return rhom

def nu_R(emu, R, z):

    M = Mass_R(emu, R)
    nu = nu_M(emu, M, z)
    return nu

def nu_M(emu, M, z):

    nu = dc/sigma_M(emu, M, z)
    return nu

# Lagrangian radius
def Radius_M(emu, M):

    #rhom = comoving_matter_density(emu)
    #radius = (3.*M/(4.*np.pi*rhom))**(1./3.)
    Om_m = emu.cosmo.get_Omega0()
    radius = cosmo.Radius_M(M, Om_m)
    return radius

# Virial radius
def virial_radius_M(emu, M):
    from mead_special_functions import cbrt
    return Radius_M(emu, M)/cbrt(Dv)

def Mass_R(emu, R):

    #rhom = comoving_matter_density(emu)
    #Mass = 4.*np.pi*(R**3)*rhom/3.
    Om_m = emu.cosmo.get_Omega0()
    Mass = cosmo.Mass_R(R, Om_m)
    return Mass

def Mass_nu(emu, nu, z):

    # Import
    from scipy.interpolate import InterpolatedUnivariateSpline as ius

    # Options
    log_interp = log_interp_sigma # Should sigma(M) be interpolated logarithmically?
    
    # Get internal M vs sigma arrays
    Ms_internal = emu.massfunc.Mlist
    sig0s_internal = emu.massfunc.sigs0
    sigs_internal = sig0s_internal*emu.Dgrowth_from_z(z)
    nus_internal = dc/sigs_internal 

    # Make an interpolator for sigma(M)  
    if log_interp:
        mass_interpolator = ius(nus_internal, np.log(Ms_internal))
    else:
        mass_interpolator = ius(nus_internal, Ms_internal)

    # Get sigma(M) from the interpolator at the desired masses
    if log_interp:
        Mass = np.exp(mass_interpolator(nu))
    else:
        Mass = mass_interpolator(nu)

    return Mass

def sigma_R(emu, R, z):

    M = Mass_R(emu, R)
    sigma = sigma_M(emu, M, z)
    return sigma

def sigma_M(emu, M, z):

    # TODO: This creates the interpolator AND evaluates it. Could just create an interpolator...

    # Import
    from scipy.interpolate import InterpolatedUnivariateSpline as ius

    # Options
    log_interp = log_interp_sigma # Should sigma(M) be interpolated logarithmically?
    
    # Get internal M vs sigma arrays
    Ms_internal = emu.massfunc.Mlist
    sigs_internal = emu.massfunc.sigs0

    # Make an interpolator for sigma(M)  
    if log_interp:
        sigma_interpolator = ius(np.log(Ms_internal), np.log(sigs_internal), ext='extrapolate')
    else:
        sigma_interpolator = ius(Ms_internal, sigs_internal, ext='extrapolate')
    
    # Get sigma(M) from the interpolator at the desired masses
    if log_interp:
        sigma0 = np.exp(sigma_interpolator(np.log(M)))
    else:
        sigma0 = sigma_interpolator(M)

    # Growth function (g(z=0)=1)
    g = emu.Dgrowth_from_z(z) 
    sigma = g*sigma0

    # Result assuming scale-independent growth
    return sigma

# Linear halo bias: b(M, z)
# Taken from the pre-release version of Dark Quest given to me by Takahiro
# I am not sure why this functionality was omitted from the final version
def get_bias_mass(emu, M, redshift):
    Mp = M * 1.01
    Mm = M * 0.99
    logdensp = np.log10(emu.mass_to_dens(Mp, redshift))
    logdensm = np.log10(emu.mass_to_dens(Mm, redshift))
    bp = emu.get_bias(logdensp, redshift)
    bm = emu.get_bias(logdensm, redshift)
    return (bm * 10**logdensm - bp * 10**logdensp) / (10**logdensm - 10**logdensp)

# Return an array of n(M) (dn/dM in Dark Quest notation) at user-specified halo masses
# Extrapolates if necessary, which is perhaps dangerous
def get_dndM_mass(emu, Ms, z):

    # Imports
    from scipy.interpolate import InterpolatedUnivariateSpline as ius

    # Construct an interpolator for n(M) (or dn/dM) from the emulator internals
    Ms = emu.massfunc.Mlist
    dndM = emu.massfunc.get_dndM(z)
    dndM_interp = ius(np.log(Ms), np.log(dndM), ext='extrapolate')

    # Evaluate the interpolator at the desired mass points
    return np.exp(dndM_interp(np.log(Ms)))

# Calculate the number density of haloes in the range Mmin to Mmax
# Result is [(Mpc/h)^-3]
def ndenshalo(emu, Mmin, Mmax, z):

    # Parameters
    vol = 1.

    return emu.get_nhalo(Mmin, Mmax, vol, z)

# Calculate the average mass between two limits, weighted by the halo mass function
# Result is [Msun/h]
def mass_avg(emu, Mmin, Mmax, z):

    # Import
    from scipy.interpolate import InterpolatedUnivariateSpline as ius
    from scipy.integrate import quad

    # Parameters
    epsabs = 1e-5 # Integration accuracy

    # Construct an interpolator for n(M) (or dn/dM) from the emulator internals
    Ms = emu.massfunc.Mlist
    dndM = emu.massfunc.get_dndM(z)
    log_dndM_interp = ius(np.log(Ms), np.log(dndM), ext='extrapolate')

    # Number density of haloes in the mass range
    n = ndenshalo(emu, Mmin, Mmax, z) 

    # Integrate to get the average mass
    Mav, _ = quad(lambda M: M*np.exp(log_dndM_interp(np.log(M))), Mmin, Mmax, epsabs=epsabs)

    return Mav/n

# Averages the halo-halo correlation function over mass ranges to return the weighted by mass function mean version
def get_xiauto_mass_avg(emu, rs, M1min, M1max, M2min, M2max, z):

    # Import
    from scipy.interpolate import InterpolatedUnivariateSpline as ius
    from scipy.interpolate import RectBivariateSpline as rbs
    from scipy.integrate import dblquad

    # Parameters
    epsabs = 1e-3 # Integration accuracy
    nM = 6        # Number of halo-mass bins in each of M1 and M2 directions

    # Calculations
    nr = len(rs)

    # Number densities of haloes in each sample
    n1 = ndenshalo(emu, M1min, M1max, z)
    n2 = ndenshalo(emu, M2min, M2max, z)

    # Arrays for halo masses
    M1s = mead.logspace(M1min, M1max, nM)
    M2s = mead.logspace(M2min, M2max, nM)
    
    # Get mass function interpolation
    Ms = emu.massfunc.Mlist
    dndM = emu.massfunc.get_dndM(z)
    log_dndM_interp = ius(np.log(Ms), np.log(dndM))

    # Loop over radii
    xiauto_avg = np.zeros((nr))
    for ir, r in enumerate(rs):

        # Get correlation function interpolation
        # Note that this is not necessarily symmetric because M1, M2 run over different ranges
        xiauto_mass = np.zeros((nM, nM))
        for iM1, M1 in enumerate(M1s):
            for iM2, M2 in enumerate(M2s):
                xiauto_mass[iM1, iM2] = emu.get_xiauto_mass(r, M1, M2, z)
        xiauto_interp = rbs(np.log(M1s), np.log(M2s), xiauto_mass)

        # Integrate interpolated functions
        xiauto_avg[ir], _ = dblquad(lambda M1, M2: xiauto_interp(np.log(M1),np.log(M2))*np.exp(log_dndM_interp(np.log(M1))+log_dndM_interp(np.log(M2))),
                                    M1min, M1max,
                                    lambda M1: M2min, lambda M1: M2max,
                                    epsabs=epsabs)

    return xiauto_avg/(n1*n2)

def linear_halo_bias(emu, M, z, klin, Pk_klin):

    # Source of linear halo bias
    # ibias = 1: Linear bias from emulator
    # ibias = 2: Linear bias from halo-halo spectrum at large wavenumber
    # ibias = 3: Linear bias from halo-matter spectrum at large wavenumber
    ibias = 2

    if ibias == 1:
        b = get_bias_mass(emu, M, z)[0]
    elif ibias == 2:
        b = np.sqrt(emu.get_phh_mass(klin, M, M, z)/Pk_klin)
    elif ibias == 3:
        b = emu.get_phm_mass(klin, M, z)/Pk_klin
    else:
        raise ValueError('Linear bias recipe for beta_NL not recognised')

    return b

def R_hh(emu, ks, M1, M2, z):

    # Cross correlation coefficient between halo masses

    P12 = emu.get_phh_mass(ks, M1, M2, z)
    P11 = emu.get_phh_mass(ks, M1, M1, z)
    P22 = emu.get_phh_mass(ks, M2, M2, z)

    return P12/np.sqrt(P11*P22)

def beta_NL(emu, vars, ks, z, var='Mass'):

    # Beta_NL function

    # Force Beta_NL to zero at large scales?
    # force_BNL_zero = 0: No
    # force_BNL_zero = 1: Yes, via addative correction
    # force_BNL_zero = 2: Yes, via multiplicative correction
    force_BNL_zero = 0

    # Parameters
    klin = 0.02  # Large 'linear' scale [h/Mpc]

    # Set array name sensibly
    if var == 'Mass':
        Ms = vars
    elif var == 'Radius':
        Rs = vars
        Ms = Mass_R(emu, Rs)
    elif var == 'nu':
        nus = vars
        Ms = Mass_nu(emu, nus, z)
    else:
        raise ValueError('Error, mass variable for beta_NL not recognised')
    
    # klin must be a numpy array for this to work later
    klin = np.array([klin]) 
    
    # Linear power
    Pk_lin = emu.get_pklin_from_z(ks, z)
    Pk_klin = emu.get_pklin_from_z(klin, z)
    
    # Calculate beta_NL by looping over mass arrays
    beta = np.zeros((len(Ms), len(Ms), len(ks)))  
    for iM1, M1 in enumerate(Ms):

        # Linear halo bias
        b1 = linear_halo_bias(emu, M1, z, klin, Pk_klin)

        for iM2, M2 in enumerate(Ms):

            if iM2 >= iM1:

                # Linear halo bias
                b2 = linear_halo_bias(emu, M2, z, klin, Pk_klin)

                # Halo-halo power spectrum
                Pk_hh = emu.get_phh_mass(ks, M1, M2, z)
                    
                # Create beta_NL
                beta[iM1, iM2, :] = Pk_hh/(b1*b2*Pk_lin)-1.

                # Force Beta_NL to be zero at large scales if necessary
                if force_BNL_zero != 0:
                    Pk_hh0 = emu.get_phh_mass(klin, M1, M2, z)
                    db = Pk_hh0/(b1*b2*Pk_klin)-1.
                    if force_BNL_zero == 1:
                        beta[iM1, iM2, :] = beta[iM1, iM2, :]-db # Addative correction
                    elif force_BNL_zero == 2:
                        beta[iM1, iM2, :] = (beta[iM1, iM2, :]+1.)/(db+1.)-1. # Multiplicative correction
                    else:
                        raise ValueError('force_BNL_zero not set correctly')

            else:

                # Use symmetry to not double calculate
                beta[iM1, iM2, :] = beta[iM2, iM1, :]
         
    return beta 

def calculate_rescaling_params(emu_ori, emu_tgt, z_tgt, M1_tgt, M2_tgt):
    
    R1_tgt = Radius_M(emu_tgt, M1_tgt)
    R2_tgt = Radius_M(emu_tgt, M2_tgt)

    s, sm, z = cosmo.calculate_AW10_rescaling_parameters(z_tgt, R1_tgt, R2_tgt, 
                                                         lambda Ri, zi: sigma_R(emu_ori, Ri, zi), 
                                                         lambda Ri, zi: sigma_R(emu_tgt, Ri, zi),
                                                         emu_ori.cosmo.get_Omega0(),
                                                         emu_tgt.cosmo.get_Omega0(),
                                                        )
    return s, sm, z