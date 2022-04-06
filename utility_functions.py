# Utility functions for the Halo Model calculations

import numpy as np

# Fourier transform of a tophat function.
# Parameters
xmin_Tk = 1e-5
def Tophat_k(x):
    # Normalised tophat Fourier transform function such that T(x=0)=1
    xmin = xmin_Tk
    return np.where(np.abs(x)<xmin, 1.-x**2/10., (3./x**3)*(np.sin(x)-x*np.cos(x)))


def get_sigmaR(Rs,hubble_units=True,integration_type='brute'):

	if integration_type == 'brute':
		sigmaR_arr = np.asarray([sigmaR_brute_log(R,Pk_lin,kmin=kmin,kmax=kmax,nk=nk) for R in Rs])
	elif integration_type == 'quad':

	elif integration_type == 'gauleg':
	elif integration_type == 'camb':
	else:
		print('not a recognised integration_type. Try one of the following:')
		print('brute for brute force integration')
		print('quad for the general purpose quad integration')
		print('gauleg for Gaussian Legendre integration')
	return sigmaR_arr





def sigma_integrand(k,R=1,Power_k='None'):
    return Power_k(0,k)*(k**2)*Tophat_k(k*R)**2

# Brute force integration, this is only slightly faster than using a loop
def sigmaR_brute_log(R,Power_k,kmin=0,kmax=1e5,nk=1e5):
	k_arr = np.logspace(np.log10(kmin),np.log10(kmax),int(nk))
	dln_k = np.log(k_arr[1]/k_arr[0])
	def sigma_R_vec(R):
		def sigma_integrand(k):
    		return Power_k(0,k)*(k**2)*Tophat_k(k*R)**2

    	sigmaR= np.sqrt(sum(dln_k*k_arr*sigma_integrand(k_arr,R,Power_k))/(2.*np.pi**2))
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

def sigma_R_gauleg(R, Power_k, kmin=0.0, kmax=1e5,nk=1e4, nG=20)
	
	[x,w] = roots_legendre(nG+1)

	lower_bound =kmin
	upper_bound =kmax
	nargs=int(nk)
	
	#   Do we need to define boundaries?
	#   check how the integrand looks like 
	arg = np.logspace(np.log10(lower_bound),np.log10(upper_bound),nargs)

	def sigma_integrand(k,R):
        return Power_k(0,k)*(k**2)*Tophat_k(k*R)**2

    integrand = sigma_integrand
	table  = integrand(arg,1)
	maxima = argrelextrema(table,np.greater)
	minima = argrelextrema(table,np.less)
	exterma =  np.concatenate((minima, maxima), axis=None)
	if(len(exterma)>2):
	#   do piecewise integration
	    min_arg = np.min(exterma)
	    extra_points = np.linspace(0,min_arg-1,20)
	    args_limits  = np.sort(np.concatenate((exterma, extra_points.astype(int)), axis=None))
	    integ_limits = arg[args_limits]
	    a_arr = integ_limits[:-1]
	    b_arr = integ_limits[1::]

	from scipy.special import roots_legendre
	from scipy.signal import argrelextrema

	def legendre_integral(R):
	#     get the roots and weights for a 2nG+1 polynomial defined between -1 and 1
    	def integral_a_b(a,b):
        	y_all=0.5*(b-a)*x+0.5*(b+a)
        	y=y_all[y_all>=0.]
        	return (b-a)*0.5*sum(w[y_all>=0.]*integrand(y,R))
    	integral_func = np.vectorize(integral_a_b)
    	return sum(integral_func(a_arr/R,b_arr/R))

    sigma_func = np.vectorize(legendre_integral, excluded=['Power_k']) 
    return sigma_func(R)


# kmin=1e-3
# kmax=1e4
# nk=1e5
# sigmaR_b = sigmaR_brute_log(R,Pk_lin,kmin=kmin,kmax=kmax,nk=nk)

# sigmaR_b = np.asarray([sigmaR_brute_log(R,Pk_lin,kmin=kmin,kmax=kmax,nk=nk) for R in Rs])

kmin=1e-5
kmax=1e5
nk=1e4
lower_bound =kmin
upper_bound =kmax
nargs=nk
nG=20
[x,w] = roots_legendre(nG+1)
#   Do we need to define boundaries?
#   check how the integrand looks like 
arg = np.logspace(np.log10(lower_bound),np.log10(upper_bound),int(nargs))
integrand = sigma_integrand
table  = integrand(arg,1,Power_k=Pk_lin)
maxima = argrelextrema(table,np.greater)
minima = argrelextrema(table,np.less)
exterma =  np.concatenate((minima, maxima), axis=None)
if(len(exterma)>2):
#   do piecewise integration
    min_arg = np.min(exterma)
    extra_points = np.linspace(0,min_arg-1,20)
    args_limits  = np.sort(np.concatenate((exterma, extra_points.astype(int)), axis=None))
    integ_limits = arg[args_limits]
    a_arr = integ_limits[:-1]
    b_arr = integ_limits[1::]

IntegRs = np.asarray([legendre_integral(sigma_integrand,a_arr/R,b_arr/R,R=R,Power_k=Pk_lin) for R in Rs])
sigmaR_marika = np.sqrt(IntegRs/(2.*np.pi**2))




# def sigmaR_brute_log(R,Power_k,kmin=0,kmax=1e5,nk=1e5):
#     k_arr = np.logspace(np.log10(kmin),np.log10(kmax),int(nk))
#     dln_k = np.log(k_arr[1]/k_arr[0])
#     sigmaR= np.sqrt(sum(dln_k*k_arr*sigma_integrand(k_arr,R,Power_k))/(2.*np.pi**2))
#     return sigmaR
sigmaR_mead = sigma_R(Rs,Pk_lin)