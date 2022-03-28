# Standard
import numpy as np

# Set of right-hand-side equations
def _SIR_equations(t, y, R0, G0, f, m):
    '''
    Right-hand side of the SIR equations
    Parameters:
        t (array): not necessary here, but might be for solve_ivp?
        y (3n array): Array of values for solve_ivp
        R0 (nxn array): Symmetric R0 matrix between groups
        G0 (float): Ratio of infected duration to recovered duration (usually << 1; 0 means lasting immunity)
        f (n array): Group fractions (should sum to unity)
        m (nxn array): Symmetric group mixing matrix
    '''
    # Check size of problem (n should be divisible by 3)
    n3 = len(y)
    if (n3%3 != 0):
        raise ValueError('Input array must be divisible by 3')
    ng = n3//3 # Number of groups

    # Unpack y variables
    # TODO: Replace slow loops?
    S = np.zeros(ng); I = np.zeros(ng); R = np.zeros(ng)
    for ig in range(ng):
        S[ig] = y[0*(ng-1)+ig+0]
        I[ig] = y[1*(ng-1)+ig+1]
        R[ig] = y[2*(ng-1)+ig+2] 
    
    # Equations
    # TODO: Replce slow loops?
    G = G0*R
    T = np.zeros(ng)
    for i in range(ng):
        for j in range(ng):
            T[i] = T[i]+I[j]*f[j]*R0[j,i]*m[j,i]/f[i]
        T[i] = T[i]*S[i]
    
    # Differential equations
    dS = -T+G
    dI = T-I
    dR = I-G

    # Return a list
    return dS.tolist()+dI.tolist()+dR.tolist()

# Set of right-hand-side equations
def _basic_SIR_equations(t, y, R0, G0=0., Vt=(lambda t, R: 0.)):
    '''
    Right-hand side of the SIR equations
    @params
        t - time array
        y - Array of values for solve_ivp
        R0t - R0 function
        G0 - Ratio of infected duration to recovered duration (usually << 1; 0 means lasting immunity)
    '''
    # Unpack y variables
    S = y[0]; I = y[1]; R = y[2]
    
    # Differential equations
    if callable(R0):
        dS = -R0(t)*I*S+G0*R-Vt(t, R)
        dI = R0(t)*I*S-I
        dR = I-G0*R+Vt(t, R)
    else:
        dS = -R0*I*S+G0*R-Vt(t, R)
        dI = R0*I*S-I
        dR = I-G0*R+Vt(t, R)

    # Return a list
    return [dS, dI, dR]

def solve_basic_SIR(t, Ii, Ri, R0t, G0=0., Vt=(lambda t, R: 0.)):
    '''
    Routine to solve the SIR equations
    @parmas
        t - Array of time values for solution [typical infectiousness duration]
        Ii - Array of initial infected fraction for each group
        Ri - Array of initial recovered fraction for each group (array of zeros for new diseases)
        Rt - R0(t) function
        G0 - Ratio of infected duration to recovered duration (usually << 1; 0 means immunity is forever)
        Vt - Vaccination rate (fraction of population per infectiouness time)
    '''
    from scipy.integrate import solve_ivp

    # Calculate inital suceptible fraction
    Si = 1.-Ii-Ri

    # Check that initial conditions are sensible
    for (Xi, name) in [(Si, 'susceptible'), (Ii, 'infected'), (Ri, 'recovered')]:
        if (Xi < 0. or Xi > 1.):
            raise ValueError('Initial '+name+' fraction should be between zero and one')

    # Run the ODE solver
    solution = solve_ivp(_basic_SIR_equations, (t[0], t[-1]), [Si, Ii, Ri], 
                            method='RK45', 
                            t_eval=t, 
                            args=(R0t, G0, Vt), 
                        )

    # Break solution down into SIR
    S = solution.y[0]; I = solution.y[1]; R = solution.y[2]
    return (t, S, I, R)

def solve_SIR(t, Ii, Ri, R0_matrix, G0, group_fracs, mixing_matrix):
    '''
    Routine to solve the SIR equations
    @params
        t - Array of time values for solution [typical infectiousness duration]
        Ii - Array of initial infected fraction for each group
        Ri - Array of initial recovered fraction for each group (array of zeros for new diseases)
        R0_matrix - Interaction matrix of R0 between each group
        G0 - Ratio of infected duration to recovered duration (usually << 1)
        group_fracs - Fraction of people in each group
        mixing_matrix - Mixing matrix between each group
    '''
    from scipy.integrate import solve_ivp
    from mead_vectors import check_symmetric
    
    # Parameters
    eps = 1e-6 # Accuracy for unity checks
    
    # Array sizes
    ng = len(Ii) # Number of groups
    nt = len(t)  # Number of time steps

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
    Si = 1.-Ii-Ri

    # Check that initial conditions are sensible
    for (Xi, name) in [(Si, 'susceptible'), (Ii, 'infected'), (Ri, 'recovered')]:
        if (Xi.any() < 0. or Xi.any() > 1.):
            raise ValueError('Initial '+name+' fraction should be between zero and one')

    # Run the ODE solver
    solution = solve_ivp(_SIR_equations, (t[0], t[-1]), Si.tolist()+Ii.tolist()+Ri.tolist(), 
                            method='RK45', 
                            t_eval=t, 
                            args=(R0_matrix, G0, group_fracs, mixing_matrix), 
                        )

    # Break solution down into SIR
    S = np.zeros((ng, nt)); I = np.zeros((ng, nt)); R = np.zeros((ng, nt))
    for ig in range(ng):
        S[ig, :] = solution.y[0*(ng-1)+ig+0]
        I[ig, :] = solution.y[1*(ng-1)+ig+1]
        R[ig, :] = solution.y[2*(ng-1)+ig+2]
            
    return (t, S, I, R)

def solve_discrete_SIR(Si, Ii, Ri, R0, Ti, n):
    '''
    Solves the discrete SIR model
    @params
        Si - Initial number of susceptible people
        Ii - Initial number of infected people
        Ri - Initial number of recovered (and therefore immune) people
        R0 - Average number of people an infected person infects in pool of susceptible people
        Ti - Duration of infection [days]
        n - Number of time steps [days]
    '''

    # Empty lists for solution
    S = np.zeros(n+1, dtype=int)
    I = np.zeros(n+1, dtype=int)
    R = np.zeros(n+1, dtype=int)
    
    # Empty lists for solution
    S[0] = Si; I[0] = Ii; R[0] = Ri
   
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