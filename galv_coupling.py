from numpy import diag,array,sqrt,pi,cos,angle
from numpy.linalg import eigh


from LCJ_circuit import LCJ_circuit
from basic import get_annihilation_operator, get_tmat
from units import *



"""
Module implementing a coupler circuit realizing tunable galvanic coupling between our GKP qubits. See notes for details.

Author: Liam O'Brien
"""


class galv_coupler_circuit():


	def __init__(self,L_c,C_c,J_c,phi_c,max_coupler_dimension=10,phi_grid_range=3,phi_grid_spacing=.01):

		# The + mode Hamiltonian is a harmonic oscillator
		self.H_plus = hbar/sqrt(L_c*C_c/2)*array(range(max_coupler_dimension))


		# The - mode Hamiltonian is an LCJ Hamiltonian. Construct and diagonalize it via a grid in phi
		num_grid_points = int(ceil(2*phi_grid_range/phi_grid_spacing+1))
		phi_grid = linspace(-phi_grid_range,phi_grid_range,num=num_grid_points)
		
		JJ_pot = -J_c*cos(2*pi*(phi_grid + phi_c)/flux_quantum)
		L_pot = phi_grid**2/(2*L_c)

		q_grid = -1j*hbar*(get_tmat(num_grid_points)-eye(num_grid_points))/phi_grid_spacing
		kinetic = (q_grid.conj().T @ q_grid + q_grid @ q_grid.conj().T)/C_c

		H_minus_phi_grid = diag(JJ_pot+L_pot) + kinetic
		evals_minus , evecs_minus = eigh(H_minus_phi_grid)
		evecs_minus = evecs_minus * exp(-1j*angle(evecs_minus[0,:]))	#Fix the global phase of each eigenstate to be consistent
		
		# # Fix the global phase of each eigenstate
		# for i,evec in enumerate(array(evecs_minus.T)):
		# 	positive_index = nonzero(abs(evec) > 1e-8)[0][-1] # Find index of the last entry for which the wavefunction has magnitude >= 10^-8
		# 	_sign = evec[positive_index]/abs(evec[positive_index]) # Get the sign of this entry 

		# 	# Force this entry to be real and positive
		# 	evecs_minus[:,i] = _sign*evec


		self.H_minus = evals_minus[:max_coupler_dimension]


		# Construct the phi_- and phi_+ operators in their respective eigenbases
		a_plus = get_annihilation_operator(max_coupler_dimension)
		self.phi_plus = flux_quantum*(e_charge**2*sqrt(2*L_c/C_c)/(pi*hbar))**0.25*(a_plus + a_plus.T)

		self.phi_minus = evecs_minus.conj().T[:max_coupler_dimension,:] @ diag(phi_grid) @ evecs_minus[:,:max_coupler_dimension]


		# And the q_- and q_+ operators (useful if we decide to include resistors later)
		self.q_plus = -1j*e_charge*(hbar*sqrt(C_c/(2*L_c))/(pi*e_charge**2))**0.25*(a_plus-a_plus.T)
		self.q_minus = evecs_minus.conj().T[:max_coupler_dimension,:] @ q_grid @ evecs_minus[:,:max_coupler_dimension]


	def get_Hamiltonians(self):
		return self.H_plus, self.H_minus

	def get_phis(self):
		return self.phi_plus , self.phi_minus

	def get_qs(self):
		return self.q_plus , self.q_minus




