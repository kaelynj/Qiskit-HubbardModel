# Importing standard Qiskit libraries and configuring account
from qiskit import QuantumCircuit, execute, Aer, IBMQ, BasicAer, QuantumRegister, ClassicalRegister
from qiskit.compiler import transpile, assemble
from qiskit.quantum_info import Operator, DensityMatrix
import qiskit.quantum_info as qi
from qiskit.tools.monitor import job_monitor
import random as rand
import scipy.linalg as la
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
def qc_evolve(qc, numsite, time, dt, hop, U, trotter_steps):
    #Compute angles for the onsite and hopping gates
    # based on the model parameters t, U, and dt
    #theta = hop*time/(2*trotter_steps)
    #phi = U*time/(trotter_steps)
    numq = 2*numsite
   # if np.isscalar(U):
   #     U = np.full(numsite, U)
   # if np.isscalar(hop):
   #     hop = np.full(numsite, hop)
    z_onsite = []
    x_hop = []
    y_hop = []

    #MODIFIED TO TRY SMALLER TIME STEPS
    num_steps = int(time/dt)
    theta = hop*dt/(2*trotter_steps)
    phi = U*dt/(trotter_steps)
    z_onsite.append(Operator([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, np.exp(1j*phi)]]))
    x_hop.append(Operator([[np.cos(theta), 0, 0, 1j*np.sin(theta)],
            [0, np.cos(theta), 1j*np.sin(theta), 0],
            [0, 1j*np.sin(theta), np.cos(theta), 0],
            [1j*np.sin(theta), 0, 0, np.cos(theta)]]))
    y_hop.append(Operator([[np.cos(theta), 0, 0, -1j*np.sin(theta)],
            [0, np.cos(theta), 1j*np.sin(theta), 0],
            [0, 1j*np.sin(theta), np.cos(theta), 0],
            [-1j*np.sin(theta), 0, 0, np.cos(theta)]]))

    #for step in range(num_steps):
    for trot in range(trotter_steps):

        #Onsite terms
        for i in range(0, numsite):
            qc.unitary(z_onsite[0], [i, i+numsite], label="Z_Onsite")
        qc.barrier()

        #Hopping terms
        for i in range(0,numsite-1):
            #Spin-up chain
            qc.unitary(y_hop[0], [i,i+1], label="YHop")
            qc.unitary(x_hop[0], [i,i+1], label="Xhop")
            #Spin-down chain
            qc.unitary(y_hop[0], [i+numsite, i+1+numsite], label="Xhop")
            qc.unitary(x_hop[0], [i+numsite, i+1+numsite], label="Xhop")

        qc.barrier()
    #=============================================================
    '''
    for i in range(0, numsite):
        phi = U[i]*time/(trotter_steps)
        z_onsite.append( Operator([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, np.exp(1j*phi)]]) )
        if i < numsite-1:
            theta = hop[i]*time/(2*trotter_steps)
            x_hop.append( Operator([[np.cos(theta), 0, 0, 1j*np.sin(theta)],
                          [0, np.cos(theta), 1j*np.sin(theta), 0],
                          [0, 1j*np.sin(theta), np.cos(theta), 0],
                          [1j*np.sin(theta), 0, 0, np.cos(theta)]]) )
            y_hop.append( Operator([[np.cos(theta), 0, 0, -1j*np.sin(theta)],
                         [0, np.cos(theta), 1j*np.sin(theta), 0],
                         [0, 1j*np.sin(theta), np.cos(theta), 0],
                         [-1j*np.sin(theta), 0, 0, np.cos(theta)]]))


    #Loop over each time step needed and apply onsite and hopping gates
    for trot in range(trotter_steps):
        #Onsite Terms
        for i in range(0, numsite):
            qc.unitary(z_onsite[i], [i,i+numsite], label="Z_Onsite")

            #Add barrier to separate onsite from hopping terms
            qc.barrier()

        #Hopping terms
        for i in range(0,numsite-1):
            #Spin-up chain
            qc.unitary(y_hop[i], [i,i+1], label="YHop")
            qc.unitary(x_hop[i], [i,i+1], label="Xhop")
            #Spin-down chain
            qc.unitary(y_hop[i], [i+numsite, i+1+numsite], label="Xhop")
            qc.unitary(x_hop[i], [i+numsite, i+1+numsite], label="Xhop")

            #Add barrier after finishing the time step
            qc.barrier()

      '''
#  circuit_operator = qi.Operator(qc)
#  return circuit_operator.data

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
               across each site.  Length should be numsite-1
        -U (float, list)
            Repulsion parameter of the chain.  Can be either float
               for constant repulsion or array to describe different
               repulsions for each site
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
#            qcirc.z(flip)
        #===============================================================

        qcirc.barrier()
        #Append circuit with Trotter steps needed
        qc_evolve(qcirc, nsites, t_step*dt, dt, hop, U, trotter_steps)
        #Measure the circuit
        for i in range(numq):
            qcirc.measure(i, i)

    #Choose provider and backend
        #provider = IBMQ.get_provider()
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



#================== sys_evolve_eng ==========================
'''
  Function to evolve the 1d-chain in time given a set of system parameters and using
     the qiskit qasm_simulator and compute the total energy along the way
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
               across each site.  Length should be numsite-1
        -U (float, list)
            Repulsion parameter of the chain.  Can be either float
               for constant repulsion or array to describe different
               repulsions for each site
        -trotter_steps (int)
            Number of trotter steps used to approximate the time evolution
               operator
    Outputs:
        -data (2d array of length [2*nsites, time_steps]):
            Output data of the quantum simulation.  Record the normalized counts
               for each qubit at each time step
        -energies (array of length [time_steps])
            Output data of the total energy of the system at each time step
'''

def sys_evolve_eng(nsites, excitations, total_time, dt, hop, U, trotter_steps):
    #Check for correct data types of input
    if not isinstance(nsites, int):
        raise TypeError("Number of sites should be int")
    if np.isscalar(excitations):
        raise TypeError("Initial state should be list or numpy array")
    if not np.isscalar(total_time):
        raise TypeError("Evolution time should be scalar")
    if not np.isscalar(dt):
        raise TypeError("Time step should be scalar")
    if not isinstance(trotter_steps, int):
        raise TypeError("Number of trotter slices should be int")

    numq = 2*nsites
    num_steps = int(total_time/dt)
    print('Num Steps: ',num_steps)
    print('Total Time: ', total_time)
    data = np.zeros((2**numq, num_steps))
    energies = np.zeros(num_steps)

    for t_step in range(0, num_steps):
        #Create circuit with t_step number of steps
        q = QuantumRegister(numq)
        c = ClassicalRegister(numq)
        qcirc = QuantumCircuit(q,c)

        #=========SET YOUR INITIAL STATE==============
          #Loop over each excitation
        for flip in excitations:
           qcirc.x(flip)
          # qcirc.h(flip)
           # qcirc.t(flip)
        #===============================================================

        qcirc.barrier()
        #Append circuit with Trotter steps needed
        qc_evolve(qcirc, nsites, t_step*dt, dt, hop, U, trotter_steps)
        #Measure the circuit
        for i in range(numq):
            qcirc.measure(i, i)

    #Choose provider and backend
        #provider = IBMQ.get_provider()
        #backend = Aer.get_backend('statevector_simulator')
        backend = Aer.get_backend('qasm_simulator')
        #backend = provider.get_backend('ibmq_qasm_simulator')
        #backend = provider.get_backend('ibmqx4')
        #backend = provider.get_backend('ibmqx2')
        #backend = provider.get_backend('ibmq_16_melbourne')
        shots = 8192
        max_credits = 10 #Max number of credits to spend on execution
        job_exp = execute(qcirc, backend=backend, shots=shots, max_credits=max_credits)
        #job_monitor(job_exp)
        result = job_exp.result()
        counts = result.get_counts(qcirc)
        #print(result.get_counts(qcirc))
        print("Job: ",t_step+1, " of ", num_steps," computing energy...")

    #Store results in data array and normalize them
        for i in range(2**numq):
            if counts.get(get_bin(i,numq)) is None:
                dat = 0
            else:
                dat = counts.get(get_bin(i,numq))
            data[i,t_step] = dat/shots

    #=======================================================
        #Compute energy of system
        #Compute repulsion energies
        repulsion_energy = measure_repulsion(U, nsites, counts, shots)

        #Compute hopping energies
        #Get list of hopping pairs
        even_pairs = []
        for i in range(0,nsites-1,2):
            #up_pair = [i, i+1]
            #dwn_pair = [i+nsites, i+nsites+1]
            even_pairs.append([i, i+1])
            even_pairs.append([i+nsites, i+nsites+1])
        odd_pairs = []
        for i in range(1,nsites-1,2):
            odd_pairs.append([i, i+1])
            odd_pairs.append([i+nsites, i+nsites+1])

        #Start with even hoppings, initialize circuit and find hopping pairs
        q = QuantumRegister(numq)
        c = ClassicalRegister(numq)
        qcirc = QuantumCircuit(q,c)
          #Loop over each excitation
        for flip in excitations:
            qcirc.x(flip)
        qcirc.barrier()
        #Append circuit with Trotter steps needed
        qc_evolve(qcirc, nsites, t_step*dt, dt, hop, U, trotter_steps)
        even_hopping = measure_hopping(hop, even_pairs, qcirc, numq)
        #===============================================================
        #Now do the same for the odd hoppings
        #Start with even hoppings, initialize circuit and find hopping pairs
        q = QuantumRegister(numq)
        c = ClassicalRegister(numq)
        qcirc = QuantumCircuit(q,c)
          #Loop over each excitation
        for flip in excitations:
            qcirc.x(flip)
        qcirc.barrier()
        #Append circuit with Trotter steps needed
        qc_evolve(qcirc, nsites, t_step*dt, dt, hop, U, trotter_steps)
        odd_hopping = measure_hopping(hop, odd_pairs, qcirc, numq)

        total_energy = repulsion_energy + even_hopping + odd_hopping
        energies[t_step] = total_energy
        print("Total Energy is: ", total_energy)
        print("Job: ",t_step+1, " of ", num_steps," complete")
    return data, energies



#================== measure_repulsion =========================
'''
   Measure the energy due to the repulsive U term in H
       Inputs:
           -U (float): Repulsion energy of system
           -num_sites (int): Number of sites in chain
           -results (qiskit counts object): Results from qiskit circuit run
           -shots (int): Number of shots from circuit run
       Outputs:
           -repulsion (float): Measures U*|a|^2|11> for each pair of modes
'''
def measure_repulsion(U, num_sites, results, shots):
    repulsion = 0.
    for state in results:
        #Adding in debug print statement
        #print(state)
        for i in range( int( len(state)/2 ) ):
            if state[i]=='1':
                if state[i+num_sites]=='1':
                    print("Measured State: ",state)
                    repulsion += U*results.get(state)/shots

    return repulsion



#================== measure_hopping =========================
'''
    Measure the hopping energy at a given time step for a given set of
      even/odd pairs.  Apply the diagonalizing circuit to each pair and measure
      the hopping as -t*( |a|^2*|01> - |b|^2*|10> )
        Inputs:
            -hopping (float):  Hopping energy
            -pairs (2d list):  List of pairs of qubits to apply diagonalizing circuit
            -circuit (qiskit circuit):  Circuit to append diagonalizing gates to
            -num_qubits (int): Number of qubits

        Outputs:
            -hop_eng (floats): Hopping energy at a given time step
'''
def measure_hopping(hopping, pairs, circuit, num_qubits):
    #Add diagonalizing circuit
    for pair in pairs:
        circuit.cnot(pair[0],pair[1])
        circuit.ch(pair[1],pair[0])
        circuit.cnot(pair[0],pair[1])
    circuit.measure_all()
    #Run circuit
    backend = Aer.get_backend('qasm_simulator')
    shots = 8192
    max_credits = 10 #Max number of credits to spend on execution
    hop_exp = execute(circuit, backend=backend, shots=shots, max_credits=max_credits)
    job_monitor(hop_exp)
    result = hop_exp.result()
    counts = result.get_counts(circuit)
    #Compute energy
    for pair in pairs:
        hop_eng = 0.
        for state in counts:
            if state[num_qubits-1-pair[0]]=='1':
                prob_01 = counts.get(state)/shots
                for comp_state in counts:
                    if comp_state[num_qubits-1-pair[1]]=='1':
                        hop_eng += -hopping*(prob_01 - counts.get(comp_state)/shots)
    return hop_eng







'''The procedure here is, for each fermionic mode, add the probability of every state containing
that mode (at a given time step), and renormalize the data based on the total occupation of each mode.
Afterwards, plot the data as a function of time step for each mode.'''
#================== process_run ==========================
'''
  Function to process the data output from sys_evolve or sys_evolve_eng.  Will map each of the possible basis states to
   each fermionic mode in order to plot the occupation probability as a function of time.
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
            #Grab binary string in "little Endian" encoding by reversing get_bin()
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
               across each site.  Length should be numsite-1
        -U (float, list)
            Repulsion parameter of the chain.  Can be either float
               for constant repulsion or array to describe different
               repulsions for each site
        -trotter_steps (int)
            Number of trotter steps used to approximate the time evolution
               operator
    Outputs:
        -data (2d array of length [2*nsites, time_steps])
            Output data of the quantum simulation.  Record the normalized counts
               for each qubit at each time step
'''
def sys_evolve_den(nsites, excitations, total_time, dt, hop, U, trotter_steps):
    #Check for correct data types of input
    if not isinstance(nsites, int):
        raise TypeError("Number of sites should be int")
    if np.isscalar(excitations):
        raise TypeError("Initial state should be list or numpy array")
    if not np.isscalar(total_time):
        raise TypeError("Evolution time should be scalar")
    if not np.isscalar(dt):
        raise TypeError("Time step should be scalar")
    if not isinstance(trotter_steps, int):
        raise TypeError("Number of trotter slices should be int")

    numq = 2*nsites
    num_steps = int(total_time/dt)
    print('Num Steps: ',num_steps)
    print('Total Time: ', total_time)
    data = []

    for t_step in range(0, num_steps):
        #Create circuit with t_step number of steps
        q = QuantumRegister(numq)
        c = ClassicalRegister(numq)
        qcirc = QuantumCircuit(q,c)

        #=========USE THIS REGION TO SET YOUR INITIAL STATE==============
          #Loop over each excitation
        for flip in excitations:
           qcirc.x(flip)
           #qcirc.h(flip)
#            qcirc.z(flip)
        #===============================================================

        qcirc.barrier()
        #Append circuit with Trotter steps needed
        qc_evolve(qcirc, nsites, t_step*dt, dt, hop, U, trotter_steps)
        den_mtrx_obj = DensityMatrix.from_instruction(qcirc)
        den_mtrx = den_mtrx_obj.to_operator().data
        state_vector = qi.Statevector.from_instruction(qcirc)
        #data.append(state_vector.data)
        data.append(den_mtrx)

        #Measure the circuit
        for i in range(numq):
            qcirc.measure(i, i)
        '''
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
        '''
    return data
