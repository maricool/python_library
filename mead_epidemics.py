import numpy as np

import mead_general as mead

# Set of right-hand-side equations
def SIR_equations_R0s(t, y, ts, R0s, gamma):
    
    # Unpack y variable
    S = y[0]
    I = y[1]
    R = y[2]
    
    N = S + I + R
    
    if (t < ts[0]):
        R0 = R0s[0]
    elif (t < ts[1]):
        R0 = R0s[1]
    else:
        R0 = R0s[2]
   
    # Equations (note that gamma factorises out; it has units of 1/time)
    Sdot = -gamma*R0*I*S/N
    Idot = gamma*R0*I*S/N - gamma*I
    Rdot = gamma*I
   
    return [Sdot, Idot, Rdot]

# Routine to solve the SIR equations
def solve_SIR_R0s(S_initial, I_initial, R_initial, ts, R0s, gamma):

    import numpy as np
    from scipy.integrate import solve_ivp
        
    # Time span
    ti = 0.
    tf = 30.*gamma
    nt = 1024
    t = np.linspace(ti, tf, nt)
   
    # Solve ODE system
    solution = solve_ivp(SIR_equations_R0s, (ti, tf), (S_initial, I_initial, R_initial), 
                         method='RK45',
                         t_eval=t,
                         args=(ts, R0s, gamma),
                        )

    # Break solution down into SIR
    S = solution.y[0]
    I = solution.y[1]
    R = solution.y[2]
   
    return t, S, I, R

# Set of right-hand-side equations
# TODO: t not necessary, but might be for solve_ivp?
# def SIR_equations(t, y, R0, f=None, m=None):
    
#     # Check size of problem
#     # TODO: Is this necessary?
#     n = len(y)
#     if (n%3 != 0):
#         raise ValueError('Input array must be divisible by 3')
#     n = n//3 # Floor division

#     if n == 1:

#         # Unpack y variable
#         S = y[0]
#         I = y[1]
#         R = y[2] # Not needed
        
#         # Equations
#         T = R0*I*S
#         dS = -T
#         dI = T-I
#         dR = I
#         return [dS, dI, dR]

#     else:
    
#         # Allocate arrays
#         S = np.zeros(n)
#         I = np.zeros(n)
#         R = np.zeros(n)
        
#         # Unpack y variable
#         # TODO: Slow loop?
#         for i in range(n):
#             S[i] = y[0*(n-1)+i+0]
#             I[i] = y[1*(n-1)+i+1]
#             R[i] = y[2*(n-1)+i+2]   
        
#         # Equations
#         # TODO: Slow loops?
#         T = np.zeros(n)
#         for i in range(n):
#             for j in range(n):
#                 T[i] = T[i]+I[j]*f[j]*m[j,i]
#             T[i] = T[i]*R0*S[i]/f[i]
        
#         # Allocate arrays
#         dS = np.zeros(n)
#         dI = np.zeros(n)
#         dR = np.zeros(n)
        
#         # Differential equations
#         dS = -T
#         dI = T-I
#         dR = I

#         return dS.tolist()+dI.tolist()+dR.tolist()

# Routine to solve the SIR equations
# def solve_SIR(t, I_ini, R_ini, R0, group_frac=None, mixing_matrix=None):

#     from scipy.integrate import solve_ivp
    
#     # Parameters
#     eps = 1e-6 # Accuracy for unity checks
    
#     # Array sizes
#     if isinstance(I_ini, float):
#         n = 1
#     else:
#         n = len(I_ini)
#     nt = len(t)

#     if n > 1:

#         if group_frac is None:
#             raise TypeError('Group fraction must be specified')

#         if mixing_matrix is None:
#             raise TypeError('Mixing matrix must be specified')

#         # Check array sizes
#         if n != len(group_frac):
#             raise TypeError('Fraction array should be the same size as y')
#         if n != mixing_matrix.shape[0] or n != mixing_matrix.shape[1]:
#             raise TypeError('Mixing matrix should be the same size as y')  

#         # Check mixing matrix symmetry
#         for i in range(n):
#             for j in range(n):
#                 if mixing_matrix[i, j] != mixing_matrix[j, i]:
#                     raise ValueError('Mixing matrix must be symmetric')

#         # Check fractions
#         if abs(1.-sum(group_frac)) > eps:
#             raise ValueError('Group population fractions must sum to unity')

#         # Check interaction matrix
#         for i in range(n):
#             if abs(1.-sum(mixing_matrix[i, :])) > eps:
#                 raise ValueError('Interaction matrix must sum to unity')

#     # Calculate inital suceptible fraction
#     S_ini = 1.-I_ini-R_ini

#     # Check that initial conditions are sensible
#     if n == 1:

#         if (I_ini < 0. or I_ini > 1.):
#             raise ValueError('Initial infected fraction should be between zero and one')
#         if (R_ini < 0. or R_ini > 1.):
#             raise ValueError('Initial recovered fraction should be between zero and one')
#         if (S_ini < 0. or S_ini > 1.):
#             raise ValueError('Initial susceptible fraction should be between zero and one')

#     else:

#         if (S_ini.any() < 0. or S_ini.any() > 1.):
#             raise ValueError('Initial susceptible fraction should be between zero and one')
#         if (I_ini.any() < 0. or I_ini.any() > 1.):
#             raise ValueError('Initial infected fraction should be between zero and one')
#         if (R_ini.any() < 0. or R_ini.any() > 1.):
#             raise ValueError('Initial recovered fraction should be between zero and one')

#     # Solve ODE system
#     if n == 1:

#         solution = solve_ivp(SIR_equations, (t[0], t[-1]), [S_ini, I_ini, R_ini], 
#                                 method='RK45',
#                                 t_eval=t,
#                                 args=(R0,),
#                             )

#     else:

#         solution = solve_ivp(SIR_equations, (t[0], t[-1]), S_ini.tolist()+I_ini.tolist()+R_ini.tolist(), 
#                                 method='RK45',
#                                 t_eval=t,
#                                 args=(R0, group_frac, mixing_matrix),
#                             )

#     if n == 1:

#         S = solution.y[0]
#         I = solution.y[1]
#         R = solution.y[2]

#     else:

#         # Break solution down into SIR
#         S = np.zeros((n, nt))
#         I = np.zeros((n, nt))
#         R = np.zeros((n, nt))    
#         for i in range(n):
#             S[i, :] = solution.y[0*(n-1)+i+0]
#             I[i, :] = solution.y[1*(n-1)+i+1]
#             R[i, :] = solution.y[2*(n-1)+i+2]
            
#     return t, S, I, R

# Set of right-hand-side equations
# TODO: t not necessary, but might be for solve_ivp?
def SIR_equations(t, y, R0, f, m):
    
    # Check size of problem
    # TODO: Is this necessary?
    n = len(y)
    if (n%3 != 0):
        raise ValueError('Input array must be divisible by 3')
    n = n//3 # // is floor (integer) division
    
    # Allocate arrays
    S = np.zeros(n)
    I = np.zeros(n)
    R = np.zeros(n)
    
    # Unpack y variable
    # TODO: Slow loop?
    for i in range(n):
        S[i] = y[0*(n-1)+i+0]
        I[i] = y[1*(n-1)+i+1]
        R[i] = y[2*(n-1)+i+2]   
    
    # Equations
    # TODO: Slow loops?
    T = np.zeros(n)
    for i in range(n):
        for j in range(n):
            T[i] = T[i]+I[j]*f[j]*R0[j,i]*m[j,i]/f[i]
        T[i] = T[i]*S[i]
        
    # Allocate arrays
    dS = np.zeros(n)
    dI = np.zeros(n)
    dR = np.zeros(n)
    
    # Differential equations
    dS = -T
    dI = T-I
    dR = I

    return dS.tolist()+dI.tolist()+dR.tolist()

# Routine to solve the SIR equations
def solve_SIR(t, I_ini, R_ini, R0_matrix, group_fracs, mixing_matrix):

    from scipy.integrate import solve_ivp
    
    # Parameters
    eps = 1e-6 # Accuracy for unity checks
    
    # Array sizes
    n = len(I_ini)
    nt = len(t)

    # Check array sizes
    if n != len(group_fracs):
        raise TypeError('Fraction array should be the same size as y')
    if n != len(R0_matrix[0, :]) or n != len(R0_matrix[:, 0]):
        raise TypeError('R0 array should be square and the same size as y')
    if n != len(mixing_matrix[0, :]) or n != len(mixing_matrix[:, 0]):
        raise TypeError('Mixing matrix should be square and the same size as y')

    # Check matrix symmetry
    if not mead.check_symmetric(R0_matrix):
        raise ValueError('R0 matrix must be symmetric')
    if not mead.check_symmetric(mixing_matrix):
        raise ValueError('Mixing matrix must be symmetric')

    # Check interaction matrix sums to unity row/column-wise
    for i in range(n):
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

    solution = solve_ivp(SIR_equations, (t[0], t[-1]), S_ini.tolist()+I_ini.tolist()+R_ini.tolist(), 
                            method='RK45',
                            t_eval=t,
                            args=(R0_matrix, group_fracs, mixing_matrix),
                        )

    # Break solution down into SIR
    S = np.zeros((n, nt))
    I = np.zeros((n, nt))
    R = np.zeros((n, nt))    
    for i in range(n):
        S[i, :] = solution.y[0*(n-1)+i+0]
        I[i, :] = solution.y[1*(n-1)+i+1]
        R[i, :] = solution.y[2*(n-1)+i+2]
            
    return t, S, I, R

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
            
    # Return lists of values at each time step
    return S, I, R