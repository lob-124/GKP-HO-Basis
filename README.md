This repository contains code used to simulate the (open system) dynamics of a superconducting circuit that we use to encode a GKP qubit, and do gate operations on said qubit.

The basic structure of the code is as follows:

 1.**Driver files**: These files drive the simulation, by reading in command line parameters, data from input files, creating the necessary objects, calling subroutines implemented in other modules, organizing/stroing the ouput data, and writing the results of the simulations to disk.
In this repository, the driver files are indicated by the *_driver.py* prefix. Each one does a specific type of simulation:
- `gkp_dynamics_driver.py`: Drives simulations of the bare stabilization protocol (i.e., the protocol defined in [the first paper](https://arxiv.org/abs/2405.05671)), with the option to include charge/flux noise, photon loss, and mistiming of the free segment.
- `gkp_sqrtT_gate_dynamics_driver.py`: Drives simulations of the sqrt(T) gate protocol (to appear on ArXiv). The potential coefficients from the T-gate circuit elements should be read in from an input file
- `gkp_sqrtT_gate_dynamics_driver_v2.py`: Drives simulations of the sqrt(T) gate protocol (to appear on ArXiv), in the presence of imperfect control (i.e., mistiming of the protocol segments) 
 
 2. **Computation/simulation files**: These files define classes and methods implmenting key aspects of the simulation, such as the "grid" basis we use and objects for doing the open system evolution via the Stochastic Schrodinger Equation (SSE) formalism. Sepcifically:
 - `SSE_evolver.py`: A module implementing the `sse_evolver` class, which implements functions for simulating open system evolution using the SSS formalism
 - `LCJ_circuit.py`: A module implementing the `LCJ_circuit` class, which constructs matrices for operators on the Hilbert space for the circuit we're working with, and implements calculations of various observables (e.g., the Pauli operators for our qubit, the supercurrent through the Josephson junction, etc)
 - `HO_grid.py`: A module used to contruct a "grid basis", which is an (overcomplete) basis of harmonic oscillator eigenstates centered at regularly spaced points in space. Such a basis is particularly convenient for our simulations. This module is used heavily by `LCJ_circuit.py`
 -  `quarter_cycle_in_grid_basis.py`: A module that constructs the matrices for the evolution of the grid states under the LC Hamiltonian by 1/4 of an LC period. This module essentially implements an analytic formula we derived for this evolution, and is used when creating an `LCJ_circuit` object

 3. **Auxiliary files**: Additional supporting files containing useful methods for the simulation
- `basic.py`: Module implementing useful ubiquitous functions, such as the QHO wavefunctions, matrices for the annihilation operators, matrices for the derivative operator, etc
- `gaussian_bath.py`: Module implementing the `bath` class, which computes useful quantities needed to construct our master equation (among other things, the jump operators)
- `precision_math.py`: Module for doing arbitrary-precision arithmetic. This is used by the other modules when computing quantities inolvoing highly excited QHO eigenstates
- `units.py`: Module defining units for our calculations. This re-scales many of the standard SI units so that the numerical values of energy/charge/action/etc appearing in oiur simulation are of order one
 
 
