# Standard
import numpy as np

# My libraries
import mead_general as mead

# Set of right-hand-side equations
# TODO: t not necessary here, but might be for solve_ivp?
def SIR_equations(t, y, R0, f, m):
    
    # Check size of problem (n should be divisible by 3)
    n3 = len(y)
    if (n3%3 != 0):
        raise ValueError('Input array must be divisible by 3')
    ng = n3//3 # Number of groups

    # Unpack y variable
    # TODO: Slow loop?
    S = np.zeros(ng)
    I = np.zeros(ng)
    R = np.zeros(ng)
    for ig in range(ng):
        S[ig] = y[0*(ng-1)+ig+0]
        I[ig] = y[1*(ng-1)+ig+1]
        R[ig] = y[2*(ng-1)+ig+2] 
    
    # Equations
    # TODO: Slow loops?
    T = np.zeros(ng)
    for i in range(ng):
        for j in range(ng):
            T[i] = T[i]+I[j]*f[j]*R0[j,i]*m[j,i]/f[i]
        T[i] = T[i]*S[i]
    
    # Differential equations
    dS = -T
    dI = T-I
    dR = I

    # Return a list
    return dS.tolist()+dI.tolist()+dR.tolist()

# Routine to solve the SIR equations
def solve_SIR(t, I_ini, R_ini, R0_matrix, group_fracs, mixing_matrix):

    from scipy.integrate import solve_ivp
    from mead_vectors import check_symmetric
    
    # Parameters
    eps = 1e-6 # Accuracy for unity checks
    
    # Array sizes
    ng = len(I_ini) # Number of groups
    nt = len(t)     # Number of time steps

    # Check array sizes
    if ng != len(group_fracs):
        raise TypeError('Fraction array should be the same size as y')
    if ng != len(R0_matrix[0, :]) or ng != len(R0_matrix[:, 0]):
        raise TypeError('R0 array should be square and the same size as y')
    if ng != len(mixing_matrix[0, :]) or ng != len(mixing_matrix[:, 0]):
        raise TypeError('Mixing matrix should be square and the same size as y')

    # Check matrix symmetry
    if not check_symmetric(R0_matrix):
        raise ValueError('R0 matrix must be symmetric')
    if not check_symmetric(mixing_matrix):
        raise ValueError('Mixing matrix must be symmetric')

    # Check interaction matrix sums to unity row/column-wise
    for i in range(ng):
        if abs(1.-sum(mixing_matrix[i, :])) > eps:
            raise ValueError('Mixing matrix must sum to unity')

    # Check fractions
    if abs(1.-sum(group_fracs)) > eps:
        raise ValueError('Group population fractions must sum to unity')

    # Calculate inital suceptible fraction
    S_ini = 1.-I_ini-R_ini

    # Check that initial conditions are sensible      
    if (S_ini.any() < 0. or S_ini.any() > 1.):
        raise ValueError('Initial susceptible fraction should be between zero and one')
    if (I_ini.any() < 0. or I_ini.any() > 1.):
        raise ValueError('Initial infected fraction should be between zero and one')
    if (R_ini.any() < 0. or R_ini.any() > 1.):
        raise ValueError('Initial recovered fraction should be between zero and one')

    # Run the ODE solver
    solution = solve_ivp(SIR_equations, (t[0], t[-1]), S_ini.tolist()+I_ini.tolist()+R_ini.tolist(), 
                            method='RK45',
                            t_eval=t,
                            args=(R0_matrix, group_fracs, mixing_matrix),
                        )

    # Break solution down into SIR
    S = np.zeros((ng, nt))
    I = np.zeros((ng, nt))
    R = np.zeros((ng, nt))    
    for ig in range(ng):
        S[ig, :] = solution.y[0*(ng-1)+ig+0]
        I[ig, :] = solution.y[1*(ng-1)+ig+1]
        R[ig, :] = solution.y[2*(ng-1)+ig+2]
            
    return (t, S, I, R)

# Solves the discrete SIR model
# Si - Initial number of susceptible people
# Ii - Initial number of infected people
# Ri - Initial number of recovered (and therefore immune) people
# R0 - Average number of people an infected person infects
# Ti - Duration of infection [days]
# n - Number of time steps [days]
def solve_discrete_SIR(Si, Ii, Ri, R0, Ti, n):

    import numpy as np
        
    # Empty lists for solution
    S = np.zeros(n+1, dtype=int)
    I = np.zeros(n+1, dtype=int)
    R = np.zeros(n+1, dtype=int)
    
    # Empty lists for solution
    S[0] = Si
    I[0] = Ii
    R[0] = Ri
   
    # Loop over steps and update
    for i in range(n):

        N = S[i]+I[i]+R[i]
        
        new_infections = int(round(R0*I[i]*S[i]/(Ti*N)))

        if (i-Ti < 0):
            new_recoveries = 0
        else:
            new_recoveries = int(round(R0*I[i-Ti]*S[i-Ti]/(Ti*N)))

        S[i+1] = S[i]-new_infections
        I[i+1] = I[i]+new_infections-new_recoveries
        R[i+1] = R[i]+new_recoveries
            
    # Return tuple of values at each time step
    return (S, I, R)