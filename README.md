# Quantum Simulation and the Fermi-Hubbard model

## Author
  Kaelyn Ferris, 2021 @ Ohio University, Supervised by Dr. Sergio Ulloa

## Overview
Qiskit code which simulates the time evolution of an excitation in an _n_ site Fermi-Hubbard chain and plots the total occupation for each fermionic mode
as the system evolves.  We make use of a one-dimensional chain which is a much more approachable system to simulate on NISQ-era devices and simulators.

## Jupyter Notebooks
'Hubbard-Time-Evolution.ipynb' contains a derivation of the time evolution for the 3-site chain of the Fermi-Hubbard model as well as the code itself to execute the time evolution.  Towards the end of the notebook is a few example images to help clarify how the circuit is constructed and how the evolution changes as the parameters are modified.

'quantum-simulation.ipynb' contains a WIP derivation of digital quantum simulation, the Trotter-Suzuki formalism, the Fermi-Hubbard model, and the Jordan-Wigner transformation.
