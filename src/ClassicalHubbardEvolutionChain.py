import numpy as np
import scipy.linalg as la

'''
   Module to compute the time evolution of the Hubbard chain numerically.
'''

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
#Check if two states are different by a single hop
def hop(psii, psij, hopping):
    #Check spin down
    hopp = 0
    if psii[0]==psij[0]:
        #Create array of indices with nonzero values
        indi = np.nonzero(psii[1])[0]
        indj = np.nonzero(psij[1])[0]
        for i in range(len(indi)):
            if abs(indi[i]-indj[i])==1:
                hopp = -hopping
                return hopp
    #Check spin up
    if psii[1]==psij[1]:
        indi = np.nonzero(psii[0])[0]
        indj = np.nonzero(psij[0])[0]
        for i in range(len(indi)):
            if abs(indi[i]-indj[i])==1:
                hopp = -hopping
                return hopp
    return hopp

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
        for j in range(len(states)):
            psi_j = states[j]
            if j==i:
                for l in range(0,len(states[0][0])):
                    if psi_i == repel(l,psi_j):
                        H[i,j] = U
                        break
            else:
                H[i,j] = hop(psi_i, psi_j, t)
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
    excitations = 0
    for i in range(len(states[0][0])):
        if states[0][0][i]==1:
            excitations+=1
        if states[0][1][i]==1:
            excitations+=1

    mapping = get_mapping(states)

    #Initalize system
    tsteps = int(total_time/dt)
    evolve = np.zeros([tsteps, len(init_wfk)])
    mode_evolve = np.zeros([tsteps, len(mapping)])
    wfk = init_wfk
    energies = np.zeros(tsteps)

    #Store first time step in mode_evolve
    for i in range(0, len(mapping)):
        wfk_sum = 0.
        for j in mapping[i]: 
            wfk_sum += wfk[j]
        mode_evolve[0][i] = wfk_sum / excitations
        energies[0] = wfk_energy(wfk, hamiltonian)


    #Now do time evolution
    times = np.arange(0., total_time, dt)
    for t in range(1, tsteps):
        wfk = np.dot(t_operator, wfk)
        evolve[t] = np.multiply(np.conj(wfk), wfk)
        energies[t] = wfk_energy(wfk, hamiltonian)
        for i in range(0, len(mapping)):
            wfk_sum = 0.
            for j in mapping[i]:
                wfk_sum += evolve[t][j]
            mode_evolve[t][i] = wfk_sum / excitations

    #Return time evolution
    return mode_evolve, energies


