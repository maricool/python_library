import math
from mead_DarkQuest import sigma_M
import numpy as np
import scipy.integrate as integrate

import mead_general as mead
import mead_cosmology as cosmo

# Constants
Dv0 = 18.*np.pi**2 # Delta_v = ~178, EdS halo virial overdensity
dc0 = (3./20.)*(12.*np.pi)**(2./3.) # delta_c = ~1.686' EdS linear collapse threshold

# Parameters
dc_rel_tol = 1e-3 # Relative tolerance for checking 'closeness' of delta_c
Dv_rel_tol = 1e-3 # Relative tolerance for checking 'closeness' of Delta_v
Tinker_z_dep = False

# Halo-model integration scheme for integration in nu
# NOTE: Cannot use Romberg because integration is done nu, not M, and this will generally be uneven
# TODO: In HMx I use trapezoid because of fast osciallations, maybe I should also use that here
halo_integration = integrate.trapezoid
#halo_integration = integrate.simps

# W(k) integration scheme integration in r
# TODO: Add FFTlog
#win_integration = integrate.trapezoid
#win_integration = integrate.simps
win_integration = integrate.romb # Needs 2^m+1 (integer m) evenly-spaced samples in R

# Halo models for halo mass function and bias
PS = 'Press & Schecter 1974'
ST = 'Sheth & Tormen 1999'
Tinker2010 = 'Tinker 2010'

# Defaults
hm_def = Tinker2010
Dv_def = 200.  # Halo overdensity with respect to background matter density
dc_def = 1.686 # Linear collapse threshold relating nu = delta_c/sigma(M)

### Class definition ###

class halomod():

    def __init__(self, z, Om_m, hm=hm_def, Dv=Dv_def, dc=dc_def):

        # Store internal variables
        self.z = z
        self.a = 1./(1.+z)
        self.Om_m = Om_m
        self.hm = hm
        self.dc = dc
        self.Dv = Dv

        if hm == PS:
            # No initialisation required for Press & Schecter (1974)
            pass
        elif hm == ST:
            # Sheth & Tormen (1999; https://arxiv.org/abs/astro-ph/9901122) mass function parameters
            from scipy.special import gamma as Gamma
            p = 0.3
            q = 0.707
            self.p_ST = p
            self.q_ST = q
            self.A_ST = np.sqrt(2.*q)/(np.sqrt(np.pi)+Gamma(0.5-p)/2**p) # A ~ 0.21
        elif hm == Tinker2010:
            # Tinker et al. (2010; https://arxiv.org/abs/1001.3162)
            # Check Delta_v and delta_c values
            if not math.isclose(Dv, 200., rel_tol=Dv_rel_tol):
                print('Warning, Tinker (2010) only coded up for Dv=200')
            if not math.isclose(dc, 1.686, rel_tol=dc_rel_tol):
                print('Warning, dc = 1.686 assumed in Tinker (2010)')
            # Mass function from Table 4
            alpha = 0.368
            beta = 0.589
            gamma = 0.864
            phi = -0.729
            eta = -0.243
            if Tinker_z_dep:
                beta = beta*(1.+z)**0.20
                gamma = gamma*(1.+z)**-0.01
                phi = phi*(1.+z)**-0.08
                eta = eta*(1.+z)**0.27
            self.alpha_Tinker = alpha
            self.beta_Tinker = beta
            self.gamma_Tinker = gamma
            self.phi_Tinker = phi
            self.eta_Tinker = eta
            # Calibrated halo bias (not from peak-background split) from Table 2
            y = np.log10(self.Dv)
            exp = np.exp(-(4./y)**4)
            self.A_Tinker = 1.+0.24*y*exp
            self.a_Tinker = 0.44*y-0.88
            self.B_Tinker = 0.183
            self.b_Tinker = 1.5
            self.C_Tinker = 0.019+0.107*y+0.19*exp
            self.c_Tinker = 2.4
        else:
            raise ValueError('Halo model not recognised')

    def halo_mass_function(self, nu):

        '''
        Halo mass function g(nu) with nu=delta_c/sigma(M)
        Integral of g(nu) over all nu is unity
        '''
        if self.hm == PS:
            return np.sqrt(2./np.pi)*np.exp(-(nu**2)/2.)
        elif self.hm == ST:
            A = self.A_ST
            q = self.q_ST
            p = self.p_ST
            return A*(1.+((q*nu**2)**(-p)))*np.exp(-q*nu**2/2.)
        elif self.hm == Tinker2010:
            alpha = self.alpha_Tinker
            beta = self.beta_Tinker
            gamma = self.gamma_Tinker
            phi = self.phi_Tinker
            eta = self.eta_Tinker
            f1 = (1.+(beta*nu)**(-2.*phi))
            f2 = nu**(2.*eta)
            f3 = np.exp(-gamma*nu**2/2.)
            return alpha*f1*f2*f3
        else:
            raise ValueError('Halo model not recognised in halo_mass_function')

    def linear_halo_bias(self, nu):

        '''
        Halo linear bias b(nu) with nu=delta_c/sigma(M)
        Integral of b(nu)*g(nu) over all nu is unity
        '''
        if self.hm == PS:
            return 1.+(nu**2-1.)/self.dc
        elif self.hm == ST:
            p = self.p_ST
            q = self.q_ST
            return 1.+(q*(nu**2)-1.+2.*p/(1.+(q*nu**2)**p))/self.dc
        elif self.hm == Tinker2010:
            A = self.A_Tinker
            a = self.a_Tinker
            B = self.B_Tinker
            b = self.b_Tinker
            C = self.C_Tinker
            c = self.c_Tinker
            fA = A*nu**a/(nu**a+self.dc**a)
            fB = B*nu**b
            fC = C*nu**c
            return 1.-fA+fB+fC
        else:
            raise ValueError('Halo model ihm not recognised in linear_halo_bias')

### ###

### Halo model ###

def _get_nus(Ms, dc, Om_m, sigmas=None, sigma=None, Pk_lin=None):

    '''
    '''
    # Create arrays of R (Lagrangian) and nu values that correspond to the halo mass
    Rs = cosmo.Radius_M(Ms, Om_m)

    # Convert R values to nu via sigma(R)
    if sigmas is not None:
        nus = dc/sigmas # Use the provided sigma(R) values or...
    elif sigma is not None:
        nus = dc/sigma(Rs) # ...otherwise evaluate the provided sigma(R) function or...
    elif Pk_lin is not None:
        nus = cosmo.nu_R(Rs, Pk_lin, dc) # ...otherwise integrate the linear power.
    else:
        raise ValueError('Error, you need to specify (at least) one of Pk_lin, sigma or sigmas') 
    return nus

def virial_radius(M, Dv, Om_m):
    '''
    Halo virial radius based on the halo mass and overdensity condition
    '''
    return cosmo.Radius_M(M, Om_m)/np.cbrt(Dv)

def dc_NakamuraSuto(Om_mz):
    '''
    LCDM fitting function for the critical linear collapse density from Nakamura & Suto (1997; https://arxiv.org/abs/astro-ph/9612074)
    Cosmology dependence is very weak
    '''
    return dc0*(1.+0.012299*np.log10(Om_mz))

def Dv_BryanNorman(Om_mz):
    '''
    LCDM fitting function for virial overdensity from Bryan & Norman (1998; https://arxiv.org/abs/astro-ph/9710107)
    Note that here Dv is defined relative to background matter density, whereas in paper it is relative to critical density
    For Omega_m = 0.3 LCDM Dv ~ 330.
    '''
    x = Om_mz-1.
    Dv = Dv0+82.*x-39.*x**2
    return Dv/Om_mz

def mean_hm(hmod, Ms, fs, sigmas=None, sigma=None, Pk_lin=None):
    '''
    Calculate the mean of some f(M) over halo mass <f>: int f(M)n(M)dM where n(M) = dn/dM in some notations
    Note that the units of n(M) are [(Msun/h)^{-1} (Mpc/h)^{-3}] so the units of the result are [F (Mpc/h)^{-3}]
    Common: <M/rho> = 1 over all halo mass (equivalent to int g(nu)dnu = 1)
    Common: <M^2/rho> = M_NL non-linear halo mass that maximall contributes to one-halo term (not M*)
    Common: <b(M)M/rho> = 1 over all halo mass (equivalent to int g(nu)b(nu)dnu = 1)
    Common: <N(M)> with N the number of galaxies in each halo of mass M; gives mean number density of galaxies
    Common: <b(M)N(M)>/<N(M)> with N the number of galaxies in each halo of mass M; gives mean bias of galaxies

    Inputs
    hmod: halomodel class
    Ms: Array of halo masses [Msun/h]
    fs(Ms): Array of function to calculate mean density of (same length as Ms)
    sigmas(M): Optional array of previously calculated nu values corresponding to M
    sigma(R): Optional function to get sigma(R) at z of interest
    Pk_lin(k): Optional function to get linear power at z of interest
    '''
    nus = _get_nus(Ms, hmod.dc, hmod.Om_m, sigmas, sigma, Pk_lin)
    integrand = (fs/Ms)*hmod.halo_mass_function(nus)
    return halo_integration(integrand, nus)*cosmo.comoving_matter_density(hmod.Om_m)

def Pk_hm(hmod, Ms, ks, N_uv, rho_uv, Pk_lin, beta=None, sigmas=None, sigma=None, lowmass_uv=[False,False], Fourier_uv=[True,True], verbose=True):
    '''
    TODO: Remove Pk_lin dependence?
    Inputs
    hmod - halomodel class
    Ms - Array of halo masses [Msun/h]
    ks - Array of wavenumbers [h/Mpc]
    N_uv[2][Ms] - List of arrays of profile normalisations
    rho_uv[2][Ms, ks/rs] - List of array of either normalised Fourier transform of halo profile 'u' and 'v' [u(Mpc/h)^3] 
    or real-space profile from 0 to rv [u]
    Pk_lin(k) - Function to evaluate the linear power spectrum [(Mpc/h)^3]
    beta(M1, M2, k) - Optional array of beta_NL values at points Ms, Ms, ks
    sigmas(Ms) - Optional pre-computed array of linear sigma(M) values corresponding to Ms
    sigma(R) - Optional function to evaluate the linear sigma(R)
    lowmass_uv - Should a correction be made for low-mass haloes for field 'u'?
    Fourier_uv - Are haloes for field 'u' provided in Fourier space or real space?
    '''
    from time import time
    t1 = time() # Initial time

    # Create arrays of R (Lagrangian radius) and nu values that correspond to the halo mass
    nus = _get_nus(Ms, hmod.dc, hmod.Om_m, sigmas, sigma, Pk_lin)

    # Checks
    if (type(N_uv) != list) or (len(N_uv) != 2): raise TypeError('N must be list of length 2')
    if (type(rho_uv) != list) or (len(rho_uv) != 2): raise TypeError('N must be list of length 2')
    if (type(lowmass_uv) != list) or (len(lowmass_uv) != 2): raise TypeError('N must be list of length 2')
    if (type(Fourier_uv) != list) or (len(Fourier_uv) != 2): raise TypeError('N must be list of length 2')

    # Calculate the missing halo-bias from the low-mass part of the integral
    integrand = hmod.halo_mass_function(nus)*hmod.linear_halo_bias(nus)
    A = 1.-halo_integration(integrand, nus)
    if verbose:
        print('Missing halo-bias-mass from two-halo integrand:', A, '\n')

    # Calculate the halo profile Fourier transforms if necessary
    W_uv = []
    for i in [0, 1]:
        W_uv.append(np.empty_like(rho_uv[i]))
        if Fourier_uv[i]:
            W_uv[i] = np.copy(rho_uv[i])
        else:
            nr = rho_uv[i].shape[1] # nr=nk always, but I suppose it need not be
            for iM, M in enumerate(Ms):
                rv = virial_radius(M, hmod.Dv, hmod.Om_m)
                rs = np.linspace(0., rv, nr)
                W_uv[i][iM, :] = _halo_window(ks, rs, rho_uv[i][iM, :])
        for iM, _ in enumerate(Ms): # Normalise
            W_uv[i][iM, :] = N_uv[i][iM]*W_uv[i][iM, :]

    # Combine everything and return
    nk = len(ks)
    Pk_2h_array = np.zeros(nk)
    Pk_1h_array = np.zeros(nk)
    Pk_hm_array = np.zeros(nk)
    for ik, k in enumerate(ks):
        if beta is None:
            Pk_2h_array[ik] = _P_2h(hmod, Pk_lin, k, Ms, nus, [W_uv[0][:, ik], W_uv[1][:, ik]], lowmass_uv, A)
        else:
            Pk_2h_array[ik] = _P_2h(hmod, Pk_lin, k, Ms, nus, [W_uv[0][:, ik], W_uv[1][:, ik]], lowmass_uv, A, beta[:, :, ik])
        Pk_1h_array[ik] = _P_1h(hmod, Ms, nus, [W_uv[0][:, ik], W_uv[1][:, ik]])
        Pk_hm_array[ik] = Pk_2h_array[ik]+Pk_1h_array[ik]
    t2 = time() # Final time

    if verbose:  
        print('Halomodel calculation time [s]:', t2-t1, '\n')

    return (Pk_2h_array, Pk_1h_array, Pk_hm_array)

def Pk_hm_hu(hmod, M_h, Ms, ks, N_u, rho_u, Pk_lin, beta=None, sigmas=None, sigma=None, lowmass_u=False, Fourier_u=True, verbose=True):
    '''
    TODO: Remove Pk_lin dependence?
    Inputs
    hmod - halomodel class
    M_h: Halo mass [Msun/h]
    Ms: Array of halo masses [Msun/h]
    ks: Array of wavenumbers [h/Mpc]
    N_u[2][Ms]: List of arrays of profile normalisations
    rho_u[2][Ms, ks/rs]: List of array of either normalised Fourier transform of halo profile 'u' and 'v' [u(Mpc/h)^3] 
    or real-space profile from 0 to rv [u]
    Pk_lin(k): Function to evaluate the linear power spectrum [(Mpc/h)^3]
    beta(Ms, ks): Optional array of beta_NL values at points Ms, ks
    sigmas(Ms): Optional pre-computed array of linear sigma(M) values corresponding to Ms
    sigma(R): Optional function to evaluate the linear sigma(R)
    lowmass_u: Should a correction be made for low-mass haloes for field 'u'?
    Fourier_u: Are haloes for field 'u' provided in Fourier space or real space?
    '''
    from time import time
    from scipy.interpolate import interp1d
    t1 = time() # Initial time

    # Create arrays of R (Lagrangian radius) and nu values that correspond to the halo mass
    nus = _get_nus(Ms, hmod.dc, hmod.Om_m, sigmas, sigma, Pk_lin)

    # Calculate the missing halo-bias from the low-mass part of the integral
    integrand = hmod.halo_mass_function(nus)*hmod.linear_halo_bias(nus)
    A = 1.-halo_integration(integrand, nus)
    if verbose:
        print('Missing halo-bias-mass from two-halo integrand:', A, '\n')

    # Calculate the halo profile Fourier transforms if necessary
    W_u = np.empty_like(rho_u)
    if Fourier_u:
        W_u = np.copy(rho_u)
    else:
        nr = rho_u.shape[1] # nr=nk always, but I suppose it need not be
        for iM, M in enumerate(Ms):
            rv = virial_radius(M, hmod.Dv, hmod.Om_m)
            rs = np.linspace(0., rv, nr)
            W_u[iM, :] = _halo_window(ks, rs, rho_u[iM, :])
    for iM, _ in enumerate(Ms): # Normalisation
        W_u[iM, :] = N_u[iM]*W_u[iM, :]

    # Calculate nu(Mh) and W(Mh, k) by interpolating the input arrays
    # NOTE: W_h is not the halo profile, but the profile of the other thing (u) evaluated at the halo mass!
    nu_M_interp = interp1d(np.log(Ms), nus, kind='cubic')
    nu_h = nu_M_interp(np.log(M_h))
    W_h = np.empty_like(W_u[0, :])
    for ik, _ in enumerate(ks):
        WM_interp = interp1d(np.log(Ms), W_u[:, ik], kind='cubic')
        W_h[ik] = WM_interp(np.log(M_h))

    # Combine everything and return
    nk = len(ks)
    Pk_2h_array = np.zeros(nk)
    Pk_1h_array = np.zeros(nk)
    Pk_hm_array = np.zeros(nk)
    for ik, k in enumerate(ks):
        if beta is None:
            Pk_2h_array[ik] = _P_2h_hu(hmod, Pk_lin, k, Ms, nu_h, nus, W_u[:, ik], lowmass_u, A)
        else:
            Pk_2h_array[ik] = _P_2h_hu(hmod, Pk_lin, k, Ms, nu_h, nus, W_u[:, ik], lowmass_u, A, beta[:, ik])
        Pk_1h_array[ik] = W_h[ik] # Simply the halo profile at M=Mh here
        Pk_hm_array[ik] = Pk_2h_array[ik]+Pk_1h_array[ik]
    t2 = time() # Final time

    if verbose:  
        print('Halomodel calculation time [s]:', t2-t1, '\n')

    return (Pk_2h_array, Pk_1h_array, Pk_hm_array)

def _P_2h(hmod, Pk_lin, k, Ms, nus, W_uv, lowmass_uv, A, beta=None):
    '''
    Two-halo term at a specific wavenumber
    '''
    if beta is None:
        I_NL = 0.
    else:
        I_NL = _I_beta(hmod, beta, Ms, nus, W_uv, lowmass_uv, A)
    Iu = _I_2h(hmod, Ms, nus, W_uv[0], lowmass_uv[0], A)
    Iv = _I_2h(hmod, Ms, nus, W_uv[1], lowmass_uv[1], A)
    return Pk_lin(k)*(Iu*Iv+I_NL)

def _P_2h_hu(hmod, Pk_lin, k, Ms, nuh, nus, Wu, lowmass, A, beta=None):
    '''
    Two-halo term for halo-u at a specific wavenumber
    '''
    if beta is None:
        I_NL = 0.
    else:
        I_NL = _I_beta_hu(hmod, beta, Ms, nuh, nus, Wu, lowmass, A)
    Ih = hmod.linear_halo_bias(nuh) # Simply the linear bias
    Iu = _I_2h(hmod, Ms, nus, Wu, lowmass, A) # Same as for the standard two-halo term
    return Pk_lin(k)*(Ih*Iu+I_NL)

def _I_2h(hmod, Ms, nus, W, lowmass, A):
    '''
    Evaluate the integral that appears in the two-halo term
    '''
    integrand = W*hmod.linear_halo_bias(nus)*hmod.halo_mass_function(nus)/Ms
    I_2h = halo_integration(integrand, nus)
    if lowmass:
        I_2h += A*W[0]/Ms[0]
    I_2h = I_2h*cosmo.comoving_matter_density(hmod.Om_m)
    return I_2h

def _I_beta(hmod, beta, Ms, nus, Wuv, lowmass_uv, A):
    '''
    Evaluates the beta_NL double integral
    '''
    from numpy import trapz
    from mead_calculus import trapz2d
    integrand = np.zeros((len(nus), len(nus)))
    for iM1, nu1 in enumerate(nus):
        for iM2, nu2 in enumerate(nus):
            if iM2 >= iM1:
                M1 = Ms[iM1]
                M2 = Ms[iM2]
                W1 = Wuv[0][iM1]
                W2 = Wuv[1][iM2]
                g1 = hmod.halo_mass_function(nu1)
                g2 = hmod.halo_mass_function(nu2)
                b1 = hmod.linear_halo_bias(nu1)
                b2 = hmod.linear_halo_bias(nu2)
                integrand[iM1, iM2] = beta[iM1, iM2]*W1*W2*g1*g2*b1*b2/(M1*M2)
            else:
                integrand[iM1, iM2] = integrand[iM2, iM1]
    integral = trapz2d(integrand, nus, nus)
    if lowmass_uv[0] and lowmass_uv[1]:
        integral += (A**2)*Wuv[0][0]*Wuv[1][0]/Ms[0]**2
    if lowmass_uv[0]:
        integrand = np.zeros(len(nus))
        for iM, nu in enumerate(nus):
            M = Ms[iM]
            W = Wuv[1][iM]
            g = hmod.halo_mass_function(nu)
            b = hmod.linear_halo_bias(nu)
            integrand[iM] = beta[0, iM]*W*g*b/M
        integral += (A*Wuv[0][0]/Ms[0])*trapz(integrand, nus)
    if lowmass_uv[1]:
        for iM, nu in enumerate(nus):
            M = Ms[iM]
            W = Wuv[0][iM]
            g = hmod.halo_mass_function(nu)
            b = hmod.linear_halo_bias(nu)
            integrand[iM] = beta[iM, 0]*W*g*b/M
        integral += (A*Wuv[1][0]/Ms[0])*trapz(integrand, nus)
    return integral*cosmo.comoving_matter_density(hmod.Om_m)**2

def _I_beta_hu(hmod, beta, Ms, nuh, nus, Wu, lowmass, A):
    '''
    Evaluates the beta_NL integral for halo-u
    '''
    from numpy import trapz
    bh = hmod.linear_halo_bias(nuh)
    integrand = np.zeros(len(nus))
    for iM, nu in enumerate(nus):
        M = Ms[iM]
        W = Wu[iM]
        g = hmod.halo_mass_function(nu)
        b = hmod.linear_halo_bias(nu)
        integrand[iM] = beta[iM]*W*g*b/M
    integral = trapz(integrand, nus)
    if lowmass:
        integral += A*beta[0]*Wu[0]/Ms[0]
    return bh*integral*cosmo.comoving_matter_density(hmod.Om_m)

def _P_1h(hmod, Ms, nus, Wuv):
    '''
    One-halo term at a specific wavenumber
    '''
    integrand = Wuv[0]*Wuv[1]*hmod.halo_mass_function(nus)/Ms
    P_1h = halo_integration(integrand, nus)
    P_1h = P_1h*cosmo.comoving_matter_density(hmod.Om_m)
    return P_1h

### ###

### Beta_NL ###

def interpolate_beta_NL(ks, Ms, Ms_small, beta_NL_small):
    '''
    Interpolate beta_NL from a small grid to a large grid for halo-model calculations
    '''
    from scipy.interpolate import interp2d
    beta_NL = np.zeros((len(Ms), len(Ms), len(ks))) # Numpy array for output
    for ik, _ in enumerate(ks):
        beta_NL_interp = interp2d(np.log(Ms_small), np.log(Ms_small), beta_NL_small[:, :, ik], kind='linear')
        for iM1, M1 in enumerate(Ms):
            for iM2, M2 in enumerate(Ms):
                beta_NL[iM1, iM2, ik] = beta_NL_interp(np.log(M1), np.log(M2))
    return beta_NL

### ###

### Haloes and halo profiles ###

def _halo_window(ks, rs, Prho):
    '''
    Compute the halo window function given a 'density' profile Prho(r) = 4*pi*r^2*rho(r)
    TODO: This should almost certainly be done with a dedicated integration routine, FFTlog?

    ks: array of wavenumbers [h/Mpc]
    rs: array of radii usually from r=0 to r=rv [Mpc/h]
    Prho[rs]: array of Prho = 4pir^2*rho(r) values at different radii
    '''
    from scipy.integrate import trapezoid, simps, romb
    from scipy.special import spherical_jn

    # Spacing between points for Romberg integration
    #dr = (rs[-1]-rs[0])/(len(rs)-1) # Spacing for whole array
    dr = rs[1]-rs[0] # Spacing between points in r for Romberg integration (assumed even)

    # Calculate profile mean
    integrand = Prho
    if win_integration == romb:
        W0 = win_integration(integrand, dr)
    elif win_integration in [trapezoid, simps]:
        W0 = win_integration(integrand, rs)
    else:
        raise ValueError('Halo window function integration method not recognised')

    # Calculate profile Fourier transform
    W = np.empty_like(Prho)
    for ik, k in enumerate(ks):
        #integrand = np.sinc(k*rs/np.pi)*Prho # Numpy sinc function has unusual definition with pi
        integrand = spherical_jn(0, k*rs)*Prho # Scipy spherical Bessel is slightly faster
        if win_integration == romb:
            W[ik] = win_integration(integrand, dr)
        elif win_integration in [trapezoid, simps]:
            W[ik] = win_integration(integrand, rs)
        else:
            raise ValueError('Halo window function integration method not recognised')

    return W/W0

def Prho_isothermal(r, M, rv):
    '''
    Isothermal density profile multiplied by 4*pi*r^2
    '''
    return M/rv

def Prho_NFW(r, M, rv, c):
    '''
    NFW density profile multiplied by 4*pi*r^2
    '''
    rs = rv/c
    return M*r/(NFW_factor(c)*(1.+r/rs)**2*rs**2)

def Prho_UPP(r, z, M, r500, cosm):
    '''
    Universal pressure profile: UPP
    '''
    alphap = 0.12
    h = cosm.h
    def p(x):
        P0 = 6.41
        c500 = 1.81
        alpha = 1.33
        beta = 4.13
        gamma = 0.31
        y = c500*x
        f1 = y**(2.-gamma)
        f2 = (1.+y**alpha)**(beta-gamma)/alpha
        p = P0*(h/0.7)**(-3./2.)*f1*(r500/c500)**2/f2
        return p
    a = cosmo.scale_factor_z(z)
    H = cosmo.H(cosm, a)
    f1 = 1.65*(h/0.7)**2*H**(8./3.)
    f2 = (M/2.1e14)**(2./3.+alphap)
    return f1*f2*p(r/r500)*4.*np.pi

def rho_Prho(Prho, r, *args):
    '''
    Converts a Prho profile to a rho profile
    Take care evaluating this at zero (which will give infinity)
    '''
    return Prho(r, *args)/(4.*np.pi*r**2)

def rho_isothermal(r, M, rv):
    '''
    Density profile for an isothermal halo
    '''
    return rho_Prho(Prho_isothermal, r, M, rv)

def rho_NFW(r, M, rv, c):
    '''
    Density profile for an NFW halo
    '''
    return rho_Prho(Prho_NFW, r, M, rv, c)

def win_delta():
    '''
    Normalised Fourier tranform for a delta-function profile
    '''
    return 1.

def win_isothermal(k, rv):
    '''
    Normalised Fourier transform for an isothermal profile
    '''
    from scipy.special import sici
    Si, _ = sici(k*rv)
    return Si/(k*rv)

def win_NFW(k, rv, c):
    '''
    Normalised Fourier transform for an NFW profile
    '''
    from scipy.special import sici
    rs = rv/c
    kv = k*rv
    ks = k*rs
    Sisv, Cisv = sici(ks+kv)
    Sis, Cis = sici(ks)
    f1 = np.cos(ks)*(Cisv-Cis)
    f2 = np.sin(ks)*(Sisv-Sis)
    f3 = np.sin(kv)/(ks+kv)
    f4 = NFW_factor(c)
    return (f1+f2-f3)/f4

def NFW_factor(c):
    '''
    Factor from normalisation that always appears in NFW equations
    '''
    return np.log(1.+c)-c/(1.+c)

def conc_Duffy(M, z, halo_definition='M200'):
    '''
    Duffy et al (2008; 0804.2486) c(M) relation for WMAP5, See Table 1
    '''
    # Appropriate for the full (rather than relaxed) samples
    M_piv = 2e12 # Pivot mass [Msun/h]
    if halo_definition == 'M200':
        A = 10.14; B = -0.081; C = -1.01
    elif halo_definition in ['vir', 'virial']:
        A = 7.85; B = -0.081; C = -0.71
    elif halo_definition == 'M200c':
        A = 5.71; B = -0.084; C = -0.47
    else:
        raise ValueError('Halo definition not recognised')
    return A*(M/M_piv)**B*(1.+z)**C # Equation (4) in 0804.2486, parameters from 10th row of Table 1

### ###

### HOD ###

def HOD_Zheng(M, Mmin=1e12, sigma=0.15, M0=1e12, M1=1e13, alpha=1.):

    '''
    # HOD model from Zheng et al. (2005)
    '''

    from scipy.special import erf
    Nc = 0.5*(1.+erf(np.log(M/Mmin)/sigma))
    Ns = Nc*np.heaviside(M-M0, 0.5)*((M-M0)/M1)**alpha
    return Nc, Ns

### ###