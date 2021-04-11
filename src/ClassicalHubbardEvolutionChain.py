import numpy as np
import scipy.linalg as la

'''
   Module to compute the time evolution of the Hubbard chain numerically.
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

#Check if we need to add a repulsion term
def repel(l,state):
    if state[0][l]==1 and state[1][l]==1:
        return state
    else:
        return []

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

#Get list of list of indices to map each of the fermionic modes
# from the chosen set of basis states
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

#Obtain energy given wavefunction and hamiltonian
def wfk_energy(wfk, hamil):
    eng = np.dot(np.conj(wfk), np.dot(hamil, wfk))
    return eng

#Evolve a system with a given set of basis states, initial state, total time, and time step
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


