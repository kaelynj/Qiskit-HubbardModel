import numpy as np
import scipy.linalg as la


def get_bin(x, n=0):
    """
    Get the binary representation of x.
    Parameters: x (int), n (int, number of digits)"""
    binry = format(x, 'b').zfill(n)
    sup = list( reversed( binry[0:int(len(binry)/2)] ) )
    sdn = list( reversed( binry[int(len(binry)/2):len(binry)] ) )
    #sup = list(  binry[0:int(len(binry)/2)] ) 
    #sdn = list(  binry[int(len(binry)/2):len(binry)] ) 
    return format(x, 'b').zfill(n)



def get_states(nsite):
    states_list = []
    for state in range(0, 2**(2*nsite)):
        state_bin = get_bin(state, 2*nsite)
        state_list = [[],[]]
        for mode in range(0,nsite):
            state_list[0].append(int(state_bin[mode]))
            state_list[1].append(int(state_bin[mode+nsite]))
        states_list.append(state_list)
    
    return states_list
    


#====================== repel =====================
'''
   Function to check if matrix element H_ii should include U
     Inputs:
         -l (int): index of mode to examine
         
         -state (2d array): state to check if adding U is need
     Outputs:
         -[] or state: Used to compare against another state.  If
             state is returned, U should be added to H_ii matrix element.
             If empty list is returned, U should not be added.
'''
def repel(l,state):
    if state[0][l]==1 and state[1][l]==1:
        return state
    else:
        return []



#====================== hop =====================
'''
   Function to check if two states are adjacent by a 
    single hopping term.  Returns the matrix element H_ij for i!=j
     Inputs:
         -psii (2d array): The state to compare *with* (format here for a state
             is two arrays, the first for spin-up modes, and the second for spin-down)
         -psij (2d array): The state to compare *against*
         -hopping (float): Hopping energy between sites

    Outputs:
         -hopp (float): Matrix element H_ij (either 0 or -t)
'''
def hop(psii, psij, hopping):
    #Check spin down
    hopp = 0
    if psii[0]==psij[0]:
        #Create array of indices with nonzero values
        hops = []
        for site in range(len(psii[0])):
            if psii[1][site] != psij[1][site]:
                hops.append(site)
        if len(hops)==2 and np.sum(psii[1]) == np.sum(psij[1]):
            if hops[1]-hops[0]==1:
                hopp = -hopping
                return hopp
    #Check spin up
    if psii[1]==psij[1]:
        hops = []
        for site in range(len(psii[1])):
            if psii[0][site] != psij[0][site]:
                hops.append(site)
        if len(hops)==2 and np.sum(psii[0])==np.sum(psij[0]):
            if hops[1]-hops[0]==1:
                hopp = -hopping
                return hopp
    return hopp


#====================== get_hamiltonian =====================
'''
    Function to build the Hamiltonian matrix for the state and spin
       space spanned by a given set of states.
        Inputs:
            -states (3d list):  List of 2d arrays which correspond to each state
                possible within a given state/spin space.  The 1st array of a given
                state corresponds to the spin-up modes at each site and the 2nd 
                corresponds to the spin-down modes at each site.
            -t (float): Hopping energy
            -U (float): Repulsion energy (energies assume hbar=1)
'''
def get_hamiltonian(states, t, U):
    H = np.zeros((len(states),len(states)) )
    #Construct Hamiltonian matrix
    for i in range(len(states)):
        psi_i = states[i]
        for j in range(i, len(states)):
            psi_j = states[j]
            if j==i:
                for l in range(0,len(states[0][0])):
                    if psi_i == repel(l,psi_j):
                        H[i,j] = U
                        break
            else:
                H[i,j] = hop(psi_i, psi_j, t)
                H[j,i] = H[i,j]
    return H



#====================== get_mapping =====================
'''
   Function to map from original basis to a basis for each fermionic
      mode.
       Inputs:
           -states (3d array):  Basis states used to create the Hamiltonian
       Outputs:
           -mode_list (list):   Gets list of states from the basis which contribute
               to a given fermionic mode.  Used to transform between initial basis
               and basis which only contains one mode.
'''
def get_mapping(states):
    num_sites = len(states[0][0])
    mode_list = []
    for i in range(0,2*num_sites):
        index_list = []
        for state_index in range(0,len(states)):
            state = states[state_index]
        #Check spin-up modes
            if i < num_sites:
                if state[0][i]==1:
                    index_list.append(state_index)
        #Check spin-down modes
            else:
                if state[1][i-num_sites]==1:
                    index_list.append(state_index)
        if index_list:
            mode_list.append(index_list)
    return mode_list



#====================== wfk_energy =====================
'''
    Return the energy of a given wavefunction by sandwiching  |<psi|H|psi>|^2
     Inputs:
         -wfk (1d array): Wavefunction to find the energy of
         -hamil (matrix): Hamiltonian matrix used to find the energy.
'''#Obtain energy given wavefunction and hamiltonian
def wfk_energy(wfk, hamil):
    eng = np.dot(np.conj(wfk), np.dot(hamil, wfk))
    return eng


#====================== get_variance ====================
'''
    Compute variance of hamiltonian given a specific wavefunction
'''
def get_variance(wfk, h):
    h_squared = np.matmul(h, h)
    eng_squared = np.vdot(wfk, np.dot(h_squared, wfk))
    squared_eng = np.vdot(wfk, np.dot(h, wfk))
    var = np.sqrt(eng_squared - squared_eng)
    return var



#====================== sys_evolve =====================
'''
   Function to evolve a system with a given set of basis states, initial state, total time,
     and time step.
       Inputs:
            -states (3d list):  Array of 2d basis states in a given state space (number of electrons
               and total spin).  Each element contains one list of the spin-up modes, and a second list
               of the spin down modes.
            -init_wfk (1d list): Initial wavefunction to time evolve
            -total_time (float): Total time to run evolution (units of 1/hopping)
            -dt (float): Time step of the system.

        Outputs:
            -mode_evolve (2d list):  Time evolution of each fermionic mode.  Outer list contains the vector
                containing the occuptation of each fermionic mode at each time step.
            -energies (list):  Total energy of the system at each time step
'''
def sys_evolve(states, init_wfk, hopping, repulsion, total_time, dt):
    hamiltonian = get_hamiltonian(states, hopping, repulsion)
    t_operator = la.expm(-1j*hamiltonian*dt)
    wavefunctions = []
    mapping = get_mapping(states)

    #Initalize system
    tsteps = int(total_time/dt)
    evolve = np.zeros([tsteps, len(init_wfk)])
    mode_evolve = np.zeros([tsteps, len(mapping)])
    wfk = init_wfk
    wavefunctions.append(np.ndarray.tolist(wfk))
    energies = np.zeros(tsteps)

    #Store first time step in mode_evolve
    #evolve[0] = np.multiply(np.conj(wfk), wfk)
    evolve[0] = np.dot(np.conj(wfk), wfk)
    for i in range(0, len(mapping)):
        wfk_sum = 0.
        norm = 0.
        for j in mapping[i]:
            wfk_sum += evolve[0][j]
        
        mode_evolve[0][i] = wfk_sum
    energies[0] = wfk_energy(wfk, hamiltonian)
    norm = np.sum(mode_evolve[0])
    mode_evolve[0][:] = mode_evolve[0][:] / norm
        
        #print('wfk_sum: ',wfk_sum,'    norm: ',norm)
        
        #print('Variance: ',get_variance(wfk, hamiltonian) )
#Now do time evolution
    times = np.arange(0., total_time, dt)
    for t in range(1, tsteps):
        # Test out alternative approach
        ################################################
        t_operator = la.expm(-1j*hamiltonian*t*dt)
        wfk = np.dot(t_operator, init_wfk)
        ################################################
        #wfk = np.dot(t_operator, wfk)

        wavefunctions.append(np.ndarray.tolist(wfk))
        #evolve[t] = np.multiply(np.conj(wfk), wfk)
        evolve[t] = np.dot(np.conj(wfk),wfk)
        energies[t] = wfk_energy(wfk, hamiltonian)
        for i in range(0, len(mapping)):
            norm = 0.
            wfk_sum = 0.
            for j in mapping[i]:
                wfk_sum += evolve[t][j]
            mode_evolve[t][i] = wfk_sum
        norm = np.sum(mode_evolve[t])
        mode_evolve[t][:] = mode_evolve[t][:] / norm

    #Return time evolution
    return mode_evolve, energies, wavefunctions
