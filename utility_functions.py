# Utility functions for the Halo Model calculations

import numpy as np
import camb
import constants as const


# Fourier transform of a tophat function.
# Parameters
xmin_Tk = 1e-5
def Tophat_k(x):
    # Normalised tophat Fourier transform function such that T(x=0)=1
    xmin = xmin_Tk
    return np.where(np.abs(x)<xmin, 1.-x**2/10., (3./x**3)*(np.sin(x)-x*np.cos(x)))


def get_sigmaR(Rs,Pk_lin,kmin=1e-5,kmax=1e5,nk=1e5,integration_type='brute'):
    # 
    if integration_type == 'brute':
        sigmaR_arr = sigmaR_brute_log(Rs,Pk_lin,kmin=kmin,kmax=kmax,nk=nk)
    elif integration_type == 'quad':
        sigmaR_arr = sigma_R_quad(Rs,Pk_lin)
#     elif integration_type == 'gauleg':
#         sigmaR_arr = sigma_R_gauleg(Rs,Pk_lin)
    elif integration_type == 'camb':
        sigmaR_arr = sigma_R_camb(Rs,Pk_lin)
    else:
        print('not a recognised integration_type. Try one of the following:')
        print('brute for brute force integration')
        print('quad for the general purpose quad integration')
        # print('gauleg for Gaussian Legendre integration')
    return sigmaR_arr


def sigma_integrand(k,R=1,Power_k='None'):
    return Power_k(0,k)*(k**2)*Tophat_k(k*R)**2

# Brute force integration, this is only slightly faster than using a loop
def sigmaR_brute_log(R,Power_k,kmin=1e-5,kmax=1e5,nk=1e5):
	k_arr = np.logspace(np.log10(kmin),np.log10(kmax),int(nk))
	dln_k = np.log(k_arr[1]/k_arr[0])
	def sigma_R_vec(R):
		def sigma_integrand(k):
			return Power_k(0,k)*(k**2)*Tophat_k(k*R)**2

		sigmaR= np.sqrt(sum(dln_k*k_arr*sigma_integrand(k_arr))/(2.*np.pi**2))
		return sigmaR

	# Note that this is a function  
	sigma_func = np.vectorize(sigma_R_vec, excluded=['Power_k']) 

	# This is the function evaluated
	return sigma_func(R)

# Quad integration
def sigma_R_quad(R, Power_k):
    import scipy.integrate as integrate

    def sigma_R_vec(R):
        def sigma_integrand(k):
            return Power_k(0,k)*(k**2)*Tophat_k(k*R)**2

        # k range for integration (could be restricted if necessary)
        kmin = 0.
        kmax = np.inf

        # Evaluate the integral and convert to a nicer form
        sigma_squared, _ = integrate.quad(sigma_integrand, kmin, kmax)
        sigma = np.sqrt(sigma_squared/(2.*np.pi**2))
        return sigma

    # Note that this is a function  
    sigma_func = np.vectorize(sigma_R_vec, excluded=['Power_k']) 

    # This is the function evaluated
    return sigma_func(R)

# Get sigma(R) from CAMB
def sigma_R_camb(R, results, kmin=0.0, kmax=1e5):
	sigmaRs = results.get_sigmaR(R, hubble_units=True, return_R_z=False)
	sigmaRs = np.ravel(sigmaRs)
	return sigmaRs



def Radius_M(M, Om_m):
	'''
	Radius of a sphere containing mass M in a homogeneous universe
	'''
	return np.cbrt(3.*M/(4.*np.pi*comoving_matter_density(Om_m)))


def comoving_matter_density(Om_m):
	'''
	Comoving matter density, not a function of time [Msun/h / (Mpc/h)^3]
	'''
	return const.rhoc*Om_m


def scale_factor_z(z):
	'''
	Scale factor at redshift z: 1/a = 1+z
	'''
	return 1./(1.+z)


def Hubble_function(Om_r,Om_m,Om_w,Om_v, a):
	'''
	Hubble parameter, $(\dot{a}/a)^2$, normalised to unity at a=1
	'''
	H2=(Om_r*a**-4)+(Om_m*a**-3)+Om_w*X_de(a)+Om_v+(1.-Om)*a**-2
	return np.sqrt(H2)


def X_de(ide, a, w=None, wa=None,nw=None):
	'''
	Dark energy energy density, normalised to unity at a=1
	ide:
	0 - Fixed w = -1
	1 - w(a)CDM
	2 - wCDM
	5 - IDE II
	'''
	if(ide == 0):
		# Fixed w = -1, return ones
		return np.full_like(a, 1.) # Make numpy compatible
	if(ide == 1):
	# w(a)CDM
		return a**(-3.*(1.+w+wa))*np.exp(-3.*wa*(1.-a))
	elif(ide == 2):
		# wCDM same as w(a)CDM if w(a)=0
		return a**(-3.*(1.+w))
	# elif(ide == 5):
	#     f1=(a/a1)**nw+1.
	#     f2=(1./a1)**nw+1.
	#     f3=(1./a2)**nw+1.
	#     f4=(a/a2)**nw+1.
	#     return ((f1/f2)*(f3/f4))**(-6./nw)
	else:
		raise ValueError('ide not recognised')

# This is wrong needs to be fixed
# def sigma_R_gauleg(R, Power_k, kmin=0.0, kmax=1e5,nk=1e4, nG=20):
	# from scipy.special import roots_legendre
	# from scipy.signal import argrelextrema
# 	[x,w] = roots_legendre(nG+1)

# 	lower_bound =kmin
# 	upper_bound =kmax
# 	nargs=int(nk)

# 	#   Do we need to define boundaries?
# 	#   check how the integrand looks like 
# 	arg = np.logspace(np.log10(lower_bound),np.log10(upper_bound),nargs)

# 	def sigma_integrand(k):
# 	    return Power_k(0,k)*(k**2)*Tophat_k(k*R)**2

# 	integrand = sigma_integrand
# 	table  = integrand(arg,1)
# 	maxima = argrelextrema(table,np.greater)
# 	minima = argrelextrema(table,np.less)
# 	exterma =  np.concatenate((minima, maxima), axis=None)
# 	if(len(exterma)>2):
# 	#   do piecewise integration
# 	    min_arg = np.min(exterma)
# 	    extra_points = np.linspace(0,min_arg-1,20)
# 	    args_limits  = np.sort(np.concatenate((exterma, extra_points.astype(int)), axis=None))
# 	    integ_limits = arg[args_limits]
# 	    a_arr = integ_limits[:-1]
# 	    b_arr = integ_limits[1::]

# 	def legendre_integral(R):
# 	#     get the roots and weights for a 2nG+1 polynomial defined between -1 and 1
# 		def integral_a_b(a,b):
# 			y_all=0.5*(b-a)*x+0.5*(b+a)
# 			y=y_all[y_all>=0.]
# 			return (b-a)*0.5*sum(w[y_all>=0.]*integrand(y,R))
# 		integral_func = np.vectorize(integral_a_b)
# 		return sum(integral_func(a_arr/R,b_arr/R))

# 	sigma_func = np.vectorize(legendre_integral, excluded=['Power_k']) 
# 	return sigma_func(R)



