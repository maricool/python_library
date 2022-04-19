# Standard imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import scipy.integrate as integrate
from scipy.interpolate import interp1d

# My imports
import mead_constants as const
import mead_general as mead
import mead_interpolate as interpolate

# Parameters
AW10_future_punishment = 1e6

# Mead cosmology class, roughly analagous to that I use in Fortran
# TODO: Normalisation As vs. sigma_8 etc.
# TODO: How to include power spectra. Should I even do this? Probably I should use CAMB instead.
class cosmology():

    def __init__(self, Om_m=0.3, Om_b=0.05, Om_w=0.7, h=0.7, ns=0.96, As=1.97448e-9, 
                       m_nu=0., w=-1., wa=0., neff=3.046, YH=0.76, Tcmb=2.725, Nnu=3):

        # Primary parameters
        self.Om_m = Om_m
        self.Om_b = Om_b
        self.Om_w = Om_w
        self.h = h
        self.ns = ns
        self.As = As
        #self.sig8 = 0.8
        self.m_nu = m_nu
        self.w = w
        self.wa = wa
        self.neff = neff
        self.YH = YH
        self.Tcmb = Tcmb
        self.Nnu = Nnu

        # Fixed parameters
        # TODO: Unfix
        self.Om_v = 0.
        self.a1 = 1.
        self.a2 = 1.
        self.nw = 1.

        # Dark energy
        # 0 - Fixed w = -1
        # 1 - w(a)CDM
        # 2 - wCDM
        # 5 - IDE II
        self.ide = 2

        # Derived parameters 

        # Remaining matter Omegas and omegas
        self.w_m = self.Om_m*self.h**2
        self.w_b = self.Om_b*self.h**2
        self.w_nu = self.m_nu/const.nuconst
        self.Om_nu = self.w_nu/self.h**2
        self.Om_c = self.Om_m-self.Om_b-self.Om_nu
        self.w_c = self.Om_c*self.h**2
        
        # Radiation
        self.Om_r = 0. # TODO: Fix radiation properly

        # Total matter and curvature
        self.Om = self.Om_m+self.Om_w+self.Om_v+self.Om_r
        self.Om_k = 1.-self.Om

        # Initially empty interpolators
        self.r = None
        self.rp = None
        self.t = None
        self.f = None
        self.g = None

    def print(self):

        # Write primary parameters to screen
        print('Primary parameters')
        print('Omega_m: %1.4f' % (self.Om_m))
        print('Omega_b: %1.4f' % (self.Om_b))
        print('Omega_w: %1.4f' % (self.Om_w))
        print('h: %1.4f' % (self.h))
        print('ns: %1.4f' % (self.ns))
        print('As [1e9]: %1.4f' % (self.As*1e9))
        print('m_nu [eV]: %1.4f' % (self.m_nu))
        print('w: %1.4f' % (self.w))
        print('wa: %1.4f' % (self.wa))
        print('neff: %1.4f' % (self.neff))
        print('YH: %1.4f' % (self.YH))
        print('T_CMB [K]: %1.4f' % (self.Tcmb))
        print('N_nu: %d' % (self.Nnu))
        print()

        # Write derived parameters to screen
        print('Derived parameters')
        print('omega_m: %1.4f' % (self.w_m))
        print('omega_c: %1.4f' % (self.w_c))
        print('omega_b: %1.4f' % (self.w_b))
        print('omega_nu: %1.4f' % (self.w_nu))
        print('Omegea_c: %1.4f' % (self.Om_c))
        print('Omega_nu: %1.4f' % (self.Om_nu))
        print('Omega_k: %1.4f' % (self.Om_k))
        print()

    # Distance and age integrals
    def init_distances(self):

        #global r_tab, t_tab, rp_tab
        #global r, t, rp
        #global r0, t0

        # A small number
        small=0.

        # a values for interpolation
        amin = 1e-5
        amax = 1.
        na = 64
        a_tab = mead.logspace(amin, amax, na)

        ###

        # Integrand for the r(a) calculations and vectorise
        def r_integrand(a):
            return 1./(self.H(a)*a**2)
        r_integrand_vec = np.vectorize(r_integrand)

        # Function to integrate to get rp(a) (PARTICLE HORIZON) and vectorise
        def rp_integrate(a):
            rp, _ = integrate.quad(r_integrand_vec, 0., a)
            return rp
        rp_integrate_vec = np.vectorize(rp_integrate)

        # Function to integrate to get r(a) and vectorise
        def r_integrate(a):
            r, _ = integrate.quad(r_integrand_vec, a, 1.)
            return r
        r_integrate_vec = np.vectorize(r_integrate)

        ###

        # Integrand for the t(a) calculation and vectorise
        def t_integrand(a):
            return 1./(self.H(a)*a)
        t_integrand_vec = np.vectorize(t_integrand)

        # Function to integrate to get r(a) and vectorise
        def t_integrate(a):
            t, _ = integrate.quad(t_integrand_vec, 0., a)
            return t
        t_integrate_vec = np.vectorize(t_integrate)

        ###

        # Call the vectorised integration routine
        r_tab = r_integrate_vec(a_tab)
        rp_tab = rp_integrate_vec(a_tab)
        t_tab = t_integrate_vec(a_tab)

        # Add in values r(a=0) and t(a=0) if the 'nonzero' table has been used
        #r_tab=np.insert(r_tab,0,0.)
        #t_tab=np.insert(t_tab,0,0.)

        # Interpolaton function for r(a)
        r_func = interp1d(a_tab, r_tab, kind='cubic', fill_value='extrapolate') #This needs to be linear, not log
        def r_vectorize(a):
            if(a <= small):
                return rp_integrate(1.)
            elif(a > 1.):
                print('Error, r(a>1) called:', a)
                return None
            else:
                return r_func(a)
        self.r = np.vectorize(r_vectorize)

        # Interpolaton function for rp(a)
        rp_func = interpolate.log_interp1d(a_tab, rp_tab, kind='cubic', fill_value='extrapolate')
        def rp_vectorize(a):
            if(a <= small):
                return 0.
            elif(a > 1.):
                print('Error, rp(a>1) called:', a)
                return None
            else:
                return rp_func(a)
        self.rp = np.vectorize(rp_vectorize)

        # Interpolaton function for t(a)
        t_func = interpolate.log_interp1d(a_tab, t_tab, kind='cubic', fill_value='extrapolate')
        def t_vectorize(a):
            if(a <= small):
                return 0.
            elif(a > 1.):
                print('Error, t(a>1) called:', a)
                return None
            else:        
                return t_func(a)
        self.t = np.vectorize(t_vectorize)

        r0 = self.rp(1.)
        t0 = self.t(1.)
        print('Initialise_distances: Horizon size [dimensionless]:', r0)
        print('Initialise_distances: Horizon size [Mpc/h]:', const.Hdist*r0)
        print('Initialise_distances: Universe age [dimensionless]:', t0)
        print('Initialise_distances: Universe age [Gyr/h]:', const.Htime*t0)
        print('Initialise_distances: r(0):',  self.r(0.))
        print('Initialise_distances: rp(0):', self.rp(0.))
        print('Initialise_distances: t(0):',  self.t(0.))
        print()

    # Growth function
    def init_growth(self):

        print('Initialise_growth: Solving growth equations')

        # Calculate things associated with the linear growth

        amin = 1e-5
        amax = 1.
        na = 64
        a_tab = mead.logspace(amin, amax, na)

        # Set initial conditions for the ODE integration
        #d_init = a_lintab_nozero[0]
        d_init = amin
        v_init = 1.

        # Function to calculate delta'
        def dd(v, d, a):
            dd = 1.5*self.Omega_m(a)*d/a**2-(2.+self.AH(a)/self.H2(a))*v/a
            return dd

        # Function to get delta' and v' in the correct format for odeint
        # Note that it returns [v',delta'] in the 'wrong' order
        def dv(X, t):
            return [X[1], dd(X[1], X[0], t)]   

        # Use odeint to get g(a) and f(a) = d ln(g)/d ln(a)
        #gv=odeint(dv,[d_init,v_init],a_lintab_nozero)
        gv = odeint(dv, [d_init,v_init], a_tab)
        g_tab = gv[:, 0]
        f_tab = a_tab*gv[:, 1]/gv[:, 0]

        print('Initialise_growth: ODE solved')

        # Add in the values g(a=0) and f(a=0) if using the 'nonzero' tab
        #g_tab=np.insert(g_tab,0,0.)
        #f_tab=np.insert(f_tab,0,1.)

        print('Initialise_growth: Creating interpolators')

        # Create interpolation function for g(a)
        g_func = interpolate.log_interp1d(a_tab, g_tab, kind='cubic', fill_value='extrapolate')
        def g_vectorize(a):
            if(a < amin):
                return a
            elif(a > 1.):
                print('Error, g(a>1) called:', a)
                return None
            else:
                return g_func(a)
        self.g = np.vectorize(g_vectorize)

        # Create interpolation function for f(a) = dln(g)/dln(a)
        f_func = interpolate.log_interp1d(a_tab, f_tab, kind='cubic', fill_value='extrapolate')
        def f_vectorize(a):
            if(a < amin):
                return 1.
            elif(a > 1.):
                print('Error, f(a>1) called:', a)
                return None
            else:       
                return f_func(a)
        self.f = np.vectorize(f_vectorize)

        print('Initialise_growth: Interpolators done')

        # Check values
        print('Initialise_growth: g(0):', self.g(0.))
        print('Initialise_growth: g(amin):', self.g(amin))
        print('Initialise_growth: g(1):', self.g(1.))
        print('Initialise_growth: f(0):', self.f(0.))
        print('Initialise_growth: f(amin):', self.f(amin))
        print('Initialise_growth: f(1):', self.f(1.))
        print()

    ### ###

    ### Functions ###

    def H2(self, a):
        '''
        Squared Hubble parameter, $(\dot{a}/a)^2$, normalised to unity at a=1
        '''
        H2=(self.Om_r*a**-4)+(self.Om_m*a**-3)+self.Om_w*self.X_de(a)+self.Om_v+(1.-self.Om)*a**-2
        return H2

    def H(self, a):
        '''
        Hubble parameter, $\dot{a}/a$ normalised to unity at a=1
        '''
        return np.sqrt(self.H2(a))

    def AH(self, a):
        '''
        Acceleration parameter, $\ddot{a}/a$ normalised to unity at a=1
        '''
        AH=-0.5*((2.*self.Om_r*a**-4)+(self.Om_m*a**-3)+self.Om_w*(1.+3*self.w_de(a)*self.X_de(a))-2.*self.Om_v)
        return AH

    def X_de(self, a):
        '''
        Dark energy energy density, normalised to unity at a=1
        '''
        if(self.ide == 0):
            return np.full_like(a, 1.) # Make numpy compatible
        if(self.ide == 1):
            return a**(-3.*(1.+self.w+self.wa))*np.exp(-3.*self.wa*(1.-a))
        elif(self.ide == 2):
            return a**(-3.*(1.+self.w))
        elif(self.ide == 5):
            f1=(a/self.a1)**self.nw+1.
            f2=(1./self.a1)**self.nw+1.
            f3=(1./self.a2)**self.nw+1.
            f4=(a/self.a2)**self.nw+1.
            return ((f1/f2)*(f3/f4))**(-6./self.nw)
        else:
            raise ValueError('ide not recognised')

    def w_de(self, a):
        '''
        Dark energy equation of state parameter: w = p/rho
        '''
        if(self.ide == 0):
            return np.full_like(a, -1.) # Make numpy compatible
        elif(self.ide == 1):
            return self.w+(1.-a)*self.wa
        elif(self.ide == 2):
            return np.full_like(a, self.w) # Make numpy compatible
        elif(self.ide == 5):
            f1=(a/self.a1)**self.nw-1.
            f2=(a/self.a1)**self.nw+1.
            f3=(a/self.a2)**self.nw-1.
            f4=(a/self.a2)**self.nw+1.
            return -1.+f1/f2-f3/f4
        else:
            raise ValueError('ide not recognised')

    def Omega_r(self, a):
        '''
        Cosmological radiataion density as a function of 'a'
        '''
        return self.Om_r*(a**-4)/self.H2(a)
        
    def Omega_m(self, a):
        '''
        Cosmological matter density as a function of 'a'
        '''
        return self.Om_m*(a**-3)/self.H2(a)

    def Omega_w(self, a):
        '''
        Cosmological dark-energy density as a function of 'a'
        '''
        return self.Om_w*self.X_de(a)/self.H2(a)

    def Omega_v(self, a):
        '''
        Cosmological vacuumm density as a function of 'a'
        '''
        return self.Om_v/self.H2(a)

    def Omega(self, a):
        '''
        Cosmological total density as a function of 'a'
        '''
        return self.Omega_r(a)+self.Omega_m(a)+self.Omega_w(a)+self.Omega_v(a)

    def physical_critical_density(self, a):
        '''
        Physical critical density [(Msun/h)/(Mpc/h)^3]
        '''
        return const.rhoc*self.H2(a)

    def comoving_critical_density(self, a):
        '''
        Comoving critical density [(Msun/h)/(Mpc/h)^3]
        '''
        return self.physical_critical_density(a)*a**3

    def comoving_matter_density(self):
        '''
        Comoving matter density, not a function of epoch [(Msun/h)/(Mpc/h)^3]
        '''
        return comoving_matter_density(self.Om_m)

    def physical_matter_density(self, a):
        '''
        Physical matter density [(Msun/h)/(Mpc/h)^3]
        '''
        return physical_matter_density(a, self.Om_m)

    def Mass_R(self, R):
        '''
        Mass contained within a sphere of radius 'R' in a homogeneous universe
        '''
        return Mass_R(R, self.Om_m)

    def Radius_M(self, M):
        '''
        Radius of a sphere containing mass M in a homogeneous universe
        '''
        return Radius_M(M, self.Om_m)

    ### ###

    ### Plotting ###

    # Plot Omega_i(a)
    def plot_Omegas(self):

        # a range
        amin = 1e-3
        amax = 1.
        na = 129
        a_lintab = np.linspace(amin, amax, na)
        a_logtab = mead.logspace(amin, amax, na)
        
        plt.figure(1, figsize=(20, 6))

        # Omegas - Linear
        plt.subplot(122)
        plt.plot(a_logtab, self.Omega_r(a_logtab), label=r'$\Omega_r(a)$')
        plt.plot(a_logtab, self.Omega_m(a_logtab), label=r'$\Omega_m(a)$')
        plt.plot(a_logtab, self.Omega_w(a_logtab), label=r'$\Omega_w(a)$')
        plt.plot(a_logtab, self.Omega_v(a_logtab), label=r'$\Omega_v(a)$')
        plt.xlabel(r'$a$')
        plt.ylabel(r'$\Omega_i(a)$')
        plt.legend()

        # Omegas - Log
        plt.subplot(121)
        plt.semilogx(a_logtab, self.Omega_r(a_logtab), label=r'$\Omega_r(a)$')
        plt.semilogx(a_logtab, self.Omega_m(a_logtab), label=r'$\Omega_m(a)$')
        plt.semilogx(a_logtab, self.Omega_w(a_logtab), label=r'$\Omega_w(a)$')
        plt.semilogx(a_logtab, self.Omega_v(a_logtab), label=r'$\Omega_v(a)$')
        plt.xlabel(r'$a$')
        plt.ylabel(r'$\Omega_i(a)$')
        plt.legend()

        plt.figure(2, figsize=(20, 6))

        # w(a) - Linear
        plt.subplot(121)
        plt.axhline(0, c='k', ls=':')
        plt.axhline(1, c='k', ls=':')
        plt.axhline(-1, c='k', ls=':')
        plt.plot(a_lintab, self.w_de(a_lintab))
        plt.xlabel(r'$a$')
        plt.ylabel(r'$w(a)$')
        plt.ylim((-1.05, 1.05))

        # w(a) - Log
        plt.subplot(122)
        plt.axhline(0, c='k', ls=':')
        plt.axhline(1, c='k', ls=':')
        plt.axhline(-1, c='k', ls=':')
        plt.semilogx(a_logtab, self.w_de(a_logtab))
        plt.xlabel(r'$a$')
        plt.ylabel(r'$w(a)$')
        plt.ylim((-1.05, 1.05))
        
        plt.show()

    # Plot cosmic distances and times (dimensionless)
    # TODO: Split distance and time and add dimensions
    def plot_distances(self):
        
        plt.subplots(3, figsize=(20, 6))

        amin = 1e-4
        amax = 1.
        na = 128
        a_lin = np.linspace(amin, amax, na)
        a_log = mead.logspace(amin, amax, na)

        # Plot cosmic distance (dimensionless) vs. a on linear scale
        plt.subplot(121)
        plt.plot(a_lin, self.r(a_lin), 'g-', label='Comoving distance')
        plt.plot(a_lin, self.rp(a_lin), 'b-', label='Particle horizon')
        plt.plot(a_lin, self.t(a_lin), 'r-', label='Age')
        plt.legend()
        plt.xlabel(r'$a$')
        plt.ylabel(r'$r(a)$ or $t(a)$')

        # Plot cosmic distance (dimensionless) vs. a on log scale
        plt.subplot(122)
        plt.loglog(a_log, self.r(a_log), 'g-', label='interpolation')
        plt.loglog(a_log, self.rp(a_log), 'b-', label='interpolation')
        plt.loglog(a_log, self.t(a_log), 'r-', label='interpolation')
        plt.xlabel(r'$a$')
        plt.ylabel(r'$r(a)$ or $t(a)$')
        plt.show()

    # Plot g(a) and f(a)
    def plot_growth(self):
        
        plt.figure(1,figsize=(20, 6))

        amin = 1e-4
        amax = 1.
        na = 128
        a_lin = np.linspace(amin, amax, na)
        a_log = mead.logspace(amin, amax, na)

        # Linear scale
        plt.subplot(121)
        plt.plot(a_lin, self.g(a_lin), 'b-', label = 'Growth function')
        plt.plot(a_lin, self.f(a_lin), 'r-', label = 'Growth rate')
        plt.legend()
        plt.xlabel(r'$a$')
        plt.xlim((0, 1.0))
        plt.ylabel(r'$g(a)$ or $f(a)$')
        plt.ylim((0, 1.05))

        # Log scale
        plt.subplot(122)
        plt.semilogx(a_log, self.g(a_log), 'b-', label=r'interpolation')
        plt.semilogx(a_log, self.f(a_log), 'r-', label=r'interpolation')
        plt.xlabel(r'$a$')
        plt.ylabel(r'$g(a)$ or $f(a)$')
        plt.ylim((0, 1.05))

        # Show the plot
        plt.show()

    ### ###
    
## Definitions of cosmological functions ##

# TODO: Should these be class methods?

# Hubble function: \dot(a)/a
#def H(cosm, a):
#    H2=(cosm.Om_r*a**-4)+(cosm.Om_m*a**-3)+cosm.Om_w*X_de(cosm, a)+cosm.Om_v+(1.-cosm.Om)*a**-2
#    return np.sqrt(H2)

# Acceleration function: \ddot(a)/a
#def AH(cosm, a):
#    AH=-0.5*((2.*cosm.Om_r*a**-4)+(cosm.Om_m*a**-3)+cosm.Om_w*(1.+3*w_de(cosm, a)*X_de(cosm, a))-2.*cosm.Om_v)
#    return AH

# Dark energy density as a function of 'a'
# def X_de(cosm, a):
#     if(cosm.ide == 0):
#         return np.full_like(a, 1.) # Make numpy compatible
#     if(cosm.ide == 1):
#         return a**(-3.*(1.+cosm.w+cosm.wa))*np.exp(-3.*cosm.wa*(1.-a))
#     elif(cosm.ide == 2):
#         return a**(-3.*(1.+cosm.w))
#     elif(cosm.ide == 5):
#         f1=(a/cosm.a1)**cosm.nw+1.
#         f2=(1./cosm.a1)**cosm.nw+1.
#         f3=(1./cosm.a2)**cosm.nw+1.
#         f4=(a/cosm.a2)**cosm.nw+1.
#         return ((f1/f2)*(f3/f4))**(-6./cosm.nw)

# Dark energy equation-of-state parameter
# def w_de(cosm, a):
#     if(cosm.ide == 0):
#         return np.full_like(a, -1.) # Make numpy compatible
#     elif(cosm.ide == 1):
#         return cosm.w+(1.-a)*cosm.wa
#     elif(cosm.ide == 2):
#         return np.full_like(a, cosm.w) # Make numpy compatible
#     elif(cosm.ide == 5):
#         f1=(a/cosm.a1)**cosm.nw-1.
#         f2=(a/cosm.a1)**cosm.nw+1.
#         f3=(a/cosm.a2)**cosm.nw-1.
#         f4=(a/cosm.a2)**cosm.nw+1.
#         return -1.+f1/f2-f3/f4
    
# Omega_r as a function of 'a'
# def Omega_r(cosm, a):
#     return cosm.Om_r*(a**-4)/H(cosm, a)**2
    
# Omega_m as a function of 'a'
# def Omega_m(cosm, a):
#     return cosm.Om_m*(a**-3)/H(cosm, a)**2

# Omega_w as a function of 'a'
# def Omega_w(cosm, a):
#     return cosm.Om_w*X_de(cosm, a)/H(cosm, a)**2

# Omega_v as a function of 'a'
# def Omega_v(cosm, a):
#     return cosm.Om_v/H(cosm, a)**2

# Total Omega as a function of 'a'
# def Omega(cosm, a):
#     return Omega_r(cosm, a)+Omega_m(cosm, a)+Omega_w(cosm, a)+Omega_v(cosm, a)

# Get P(k) from CAMB file
# def read_CAMB(fname):
#     kPk=np.loadtxt(fname)
#     k=kPk[:,0]
#     Pk=kPk[:,1]
#     return k, Pk

# def create_Pk(k_tab,Pk_tab):

#     global Pk

#     # Create P(k) interpolation function
#     Pk_func=interpolation.log_interp1d(k_tab,Pk_tab,kind='cubic')
#     def Pk_vectorize(k):
#         if(k<k_tab[0]):
#             a=np.log(Pk_tab[1]/Pk_tab[0])/np.log(k_tab[1]/k_tab[0])
#             b=np.log(Pk_tab[0])-a*np.log(k_tab[0])
#             return np.exp(a*np.log(k)+b)
#         elif(k>k_tab[-1]):
#             a=np.log(Pk_tab[-2]/Pk_tab[-1])/np.log(k_tab[-2]/k_tab[-1])
#             b=np.log(Pk_tab[-1])-a*np.log(k_tab[-1])
#             return np.exp(a*np.log(k)+b)
#         else:
#             return Pk_func(k)
#     Pk=np.vectorize(Pk_vectorize)

#     return Pk

## Generic 'cosmology' functions that should *not* take cosm class as input ##

def scale_factor_z(z):
    '''
    Scale factor at redshift z: 1/a = 1+z
    '''
    return 1./(1.+z)

def redshift_a(a):
    '''
    Redshift at scale factor a: 1/a = 1+z
    '''
    return -1.+1./a

def comoving_matter_density(Om_m):
    '''
    Comoving matter density, not a function of time [Msun/h / (Mpc/h)^3]
    '''
    return const.rhoc*Om_m

def physical_matter_density(a, Om_m):
    '''
    Physical matter density [Msun/h / (Mpc/h)^3]
    '''
    return comoving_matter_density(Om_m)*a**-3

def Mass_R(R, Om_m):
    '''
    Mass contained within a sphere of radius 'R' in a homogeneous universe
    '''
    return (4./3.)*np.pi*R**3*comoving_matter_density(Om_m)

def Radius_M(M, Om_m):
    '''
    Radius of a sphere containing mass M in a homogeneous universe
    '''
    return np.cbrt(3.*M/(4.*np.pi*comoving_matter_density(Om_m)))

def Delta2(Power_k, k):
    return Power_k(k)*k**3/(2.*np.pi**2)

def sigma_R(R, Power_k):

    def sigma_R_vec(R):

        def sigma_integrand(k):
            from mead_special_functions import Tophat_k
            return Power_k(k)*(k**2)*Tophat_k(k*R)**2

        # Evaluate the integral and convert to a nicer form
        kmin = 0.; kmax = np.inf # Integration range
        sigma_squared, _ = integrate.quad(sigma_integrand, kmin, kmax)
        sigma = np.sqrt(sigma_squared/(2.*np.pi**2))
        return sigma

    # Note that this is a function
    sigma_func = np.vectorize(sigma_R_vec, excluded=['Power_k'])

    # This is the function evaluated
    return sigma_func(R)

def dsigma_R(R, Power_k):
    '''
    Calculates d(ln sigma^2)/d(ln R) by integration
    '''
    def dsigma_R_vec(R):
        def dsigma_integrand(k):
            from mead_special_functions import Tophat_k, dTophat_k
            return Power_k(k)*(k**3)*Tophat_k(k*R)*dTophat_k(k*R)

        # Evaluate the integral and convert to a nicer form
        kmin = 0.; kmax = np.inf # Integration range
        dsigma, _ = integrate.quad(dsigma_integrand, kmin, kmax)
        dsigma = R*dsigma/(np.pi*sigma_R(R, Power_k))**2
        return dsigma

    # Note that this is a function
    dsigma_func = np.vectorize(dsigma_R_vec, excluded=['Power_k'])

    # This is the function evaluated
    return dsigma_func(R)

def nu_R(R, Power_k, dc=1.686):
    return dc/sigma_R(R, Power_k)

def nu_M(M, Power_k, Om_m, dc=1.686):
    R = Radius_M(M, Om_m)
    return nu_R(R, Power_k, dc)

def Mstar(Power_k, Om_m, dc):
    '''
    nu(Mstar) = 1
    '''
    from scipy.optimize import root_scalar as root
    M1 = 1e12; M2=1e13 # Initial guesses
    sol = root(lambda M: nu_M(M, Power_k, Om_m, dc)-1., x0=M1, x1=M2) # Root finding
    return sol.root # Return of root is a root object, so isolate the solution

def calculate_AW10_rescaling_parameters(z_tgt, R1_tgt, R2_tgt, sigma_Rz_ogn, sigma_Rz_tgt, Om_m_ogn, Om_m_tgt):

    from scipy.optimize import fmin

    def rescaling_cost_function(s, z, z_tgt, R1_tgt, R2_tgt, sigma_Rz_ogn, sigma_Rz_tgt):

        # Severely punish negative z
        if (z < 0.):
            return AW10_future_punishment

        def integrand(R):
            return (1./R)*(1.-sigma_Rz_ogn(R/s, z)/sigma_Rz_tgt(R, z_tgt))**2

        integral, _ = integrate.quad(integrand, R1_tgt, R2_tgt)
        cost = integral/np.log(R2_tgt/R1_tgt)
        return cost

    s0 = 1.
    z0 = z_tgt

    s, z = fmin(lambda x: rescaling_cost_function(x[0], x[1], z_tgt, R1_tgt, R2_tgt, sigma_Rz_ogn, sigma_Rz_tgt), [s0, z0])
    sm = (Om_m_tgt/Om_m_ogn)*s**3

    # Warning
    if z < 0.:
        print('Warning: Rescaling redshift is in the future for the original cosmology')

    return s, sm, z