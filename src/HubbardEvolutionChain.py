# Importing standard Qiskit libraries and configuring account
from qiskit import QuantumCircuit, execute, Aer, IBMQ, BasicAer, QuantumRegister, ClassicalRegister
from qiskit.compiler import transpile, assemble
from qiskit.quantum_info import Operator
from qiskit.tools.monitor import job_monitor
from qiskit.tools.jupyter import *
from qiskit.visualization import *
import random as rand
import scipy.linalg as la
provider = IBMQ.load_account()
import numpy as np

#Function to convert an integer to a binary bit string using "little Endian" encoding
# where the most significant bit is the first bit
def get_bin(x, n=0):
    """
    Get the binary representation of x.
    Parameters: x (int), n (int, number of digits)"""
    binry = format(x, 'b').zfill(n)
    sup = list( reversed( binry[0:int(len(binry)/2)] ) )
    sdn = list( reversed( binry[int(len(binry)/2):len(binry)] ) )
    return format(x, 'b').zfill(n)
    #return ''.join(sup)+''.join(sdn)



#================== qc_evolve ==========================
'''
  Function to compute the time evolution operator and append the needed gates to
   a given circuit.
    Inputs:
        -qc (qiskit circuit)
            Circuit object to append time evolution gates
        -numsite (int)
            Number of sites in the one-dimensional chain
        -time (float)
            Current time of the evolution to build the operator
        -hop (float, list)
            Hopping parameter of the chain.  Can be either float
               for constant hopping or array describing the hopping
               across each site.  Length should be numsite-1
        -U (float, list)
            Repulsion parameter of the chain.  Can be either float
               for constant repulsion or array to describe different
               repulsions for each site
        -trotter_steps (int)
            Number of trotter steps used to approximate the time evolution
               operator
    Outputs:
        -None: qc is modified and returned
'''
def qc_evolve(qc, numsite, time, hop, U, trotter_steps):
    #Compute angles for the onsite and hopping gates
    # based on the model parameters t, U, and dt
    theta = hop*time/(2*trotter_steps) 
    phi = U*time/(trotter_steps)
    numq = 2*numsite
    y_hop = Operator([[np.cos(theta), 0, 0, -1j*np.sin(theta)],
                [0, np.cos(theta), 1j*np.sin(theta), 0],
                [0, 1j*np.sin(theta), np.cos(theta), 0],
                [-1j*np.sin(theta), 0, 0, np.cos(theta)]])
    x_hop = Operator([[np.cos(theta), 0, 0, 1j*np.sin(theta)],
                [0, np.cos(theta), 1j*np.sin(theta), 0],
                [0, 1j*np.sin(theta), np.cos(theta), 0],
                [1j*np.sin(theta), 0, 0, np.cos(theta)]])
    z_onsite = Operator([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 1, 0],
               [0, 0, 0, np.exp(1j*phi)]])
    
    #Loop over each time step needed and apply onsite and hopping gates
    for trot in range(trotter_steps):
        #Onsite Terms
        for i in range(0, numsite):
            qc.unitary(z_onsite, [i,i+numsite], label="Z_Onsite")
        
            #Add barrier to separate onsite from hopping terms    
            qc.barrier()

        #Hopping terms
        for i in range(0,numsite-1):
            #Spin-up chain
            qc.unitary(y_hop, [i,i+1], label="YHop")
            qc.unitary(x_hop, [i,i+1], label="Xhop")
            #Spin-down chain
            qc.unitary(y_hop, [i+numsite, i+1+numsite], label="Xhop")
            qc.unitary(x_hop, [i+numsite, i+1+numsite], label="Xhop")

            #Add barrier after finishing the time step
            qc.barrier()
    #Measure the circuit
    for i in range(numq):
        qc.measure(i, i)
        

#================== sys_evolve ==========================
'''
  Function to evolve the 1d-chain in time given a set of system parameters and using
     the qiskit qasm_simulator (will later on add in functionality to set the backend)
    Inputs:
       -nsites (int)
           Number of sites in the chain
       -excitations (list)
           List to create initial state of the system.  The encoding here is
             the first half of the qubits are the spin-up electrons for each site
             and the second half for the spin-down electrons
       -total_time (float)
           Total time to evolve the system (units of inverse energy, 1/hop)
       -dt (float)
           Time step to evolve the system with
        -hop (float, list)
            Hopping parameter of the chain.  Can be either float
               for constant hopping or array describing the hopping
               across each site.  Length should be numsite-1 (list functionality needs to be added)
        -U (float, list)
            Repulsion parameter of the chain.  Can be either float
               for constant repulsion or array to describe different
               repulsions for each site (list functionality needs to be added)
        -trotter_steps (int)
            Number of trotter steps used to approximate the time evolution
               operator
    Outputs:
        -data (2d array of length [2*nsites, time_steps])
            Output data of the quantum simulation.  Record the normalized counts
               for each qubit at each time step
'''
def sys_evolve(nsites, excitations, total_time, dt, hop, U, trotter_steps):
    #Check for correct data types of input
    if not isinstance(nsites, int):
        raise TypeError("Number of sites should be int")
    if np.isscalar(excitations):
        raise TypeError("Initial state should be list or numpy array")
    if not np.isscalar(total_time):
        raise TypeError("Evolution time should be scalar")
    if not np.isscalar(dt):
        raise TypeError("Time step should be scalar")
    if not np.isscalar(hop):
        raise TypeError("Hopping term should be scalar")
    if not np.isscalar(U):
        raise TypeError("Repulsion term should be scalar")
    if not isinstance(trotter_steps, int):
        raise TypeError("Number of trotter slices should be int")

    numq = 2*nsites
    num_steps = int(total_time/dt)
    print('Num Steps: ',num_steps)
    print('Total Time: ', total_time)
    data = np.zeros((2**numq, num_steps))
    
    for t_step in range(0, num_steps):
        #Create circuit with t_step number of steps
        q = QuantumRegister(numq)
        c = ClassicalRegister(numq)
        qcirc = QuantumCircuit(q,c)

        #=========USE THIS REGION TO SET YOUR INITIAL STATE==============
          #Loop over each excitation 
        for flip in excitations:
            qcirc.x(flip)
        #===============================================================
    
        qcirc.barrier()
        #Append circuit with Trotter steps needed
        qc_evolve(qcirc, nsites, t_step*dt, hop, U, trotter_steps)
        
    
    #Choose provider and backend
        provider = IBMQ.get_provider()
        #backend = Aer.get_backend('statevector_simulator')
        backend = Aer.get_backend('qasm_simulator')
        #backend = provider.get_backend('ibmq_qasm_simulator')
        #backend = provider.get_backend('ibmqx4')
        #backend = provider.get_backend('ibmqx2')
        #backend = provider.get_backend('ibmq_16_melbourne')

        shots = 8192
        max_credits = 10 #Max number of credits to spend on execution
        job_exp = execute(qcirc, backend=backend, shots=shots, max_credits=max_credits)
        job_monitor(job_exp)
        result = job_exp.result()
        counts = result.get_counts(qcirc)
        print(result.get_counts(qcirc))
        print("Job: ",t_step+1, " of ", num_steps," complete.")
    
    #Store results in data array and normalize them
        for i in range(2**numq):
            if counts.get(get_bin(i,numq)) is None:
                dat = 0
            else:
                dat = counts.get(get_bin(i,numq))
            data[i,t_step] = dat/shots
    return data

'''The procedure here is, for each fermionic mode, add the probability of every state containing
that mode (at a given time step), and renormalize the data based on the total occupation of each mode.
Afterwards, plot the data as a function of time step for each mode.'''
#================== process_run ==========================
'''
  Function to evolve the 1d-chain in time given a set of system parameters and using
     the qiskit qasm_simulator (will later on add in functionality to set the backend)
    Inputs:
        -num_sites (int)
           Number of sites in the chain
        -time_steps (int)
           Number of time steps in the evolution
        -dt (float)
           Time step size (units of inverse energy)
        -results (output of sys_evolve)
           List obtained from the sys_evolve function
    Outputs:
        -proc_data (2d array of size [2*num_sites, time_steps])
           Processes the data by mapping the outputs of each qubit
             into occupation of each fermionic mode of the system.
             Does this by adding and renormalizing each possible state
               into a given fermionic mode.
'''
def process_run(num_sites, time_steps, dt, results):
    proc_data = np.zeros((2*num_sites, time_steps))
    timesq = np.arange(0.,time_steps*dt, dt)

    #Sum over time steps
    for t in range(time_steps):
        #Sum over all possible states of computer
        for i in range(2**(2*num_sites)):
            #num = get_bin(i, 2*nsite)
            num = ''.join( list( reversed(get_bin(i,2*num_sites)) ) )
            #For each state, check which mode(s) it contains and add them
            for mode in range(len(num)):
                if num[mode]=='1':
                    proc_data[mode,t] += results[i,t]
    
        #Renormalize these sums so that the total occupation of the modes is 1
        norm = 0.0
        for mode in range(len(num)):
            norm += proc_data[mode,t]
        proc_data[:,t] = proc_data[:,t] / norm
    '''
    At this point, proc_data is a 2d array containing the occupation 
    of each mode, for every time step
    '''
    return proc_data
