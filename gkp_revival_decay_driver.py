from basic             import *
from units             import *
from numpy             import *
from numpy.linalg      import *
from matplotlib.pyplot import *

from numpy.random      import default_rng 
from gaussian_bath     import bath, get_J_ohmic
from LCJ_circuit       import LCJ_circuit
from SSE_evolver       import sse_evolver

from numba import njit

from struct import pack
from secrets import randbits


if __name__ == "__main__":
	from sys import argv
	#from time import perf_counter

	if len(argv) not in [16,17,18]:
		print("Usage: omega E_J/h nu Gamma gamma_q Temp Lambda N_samples N_periods N_revivals N_wells N_rungs drive_resolution_order <LCJ_save_path> <data_save_path> seed_SSE (op) seed_init (op)")
		print("Units: \n -> omega in GHz \n -> E_J in GHz \n -> Gamma in GHz \n -> gamma_q in e^2/THz \n -> Temp in Kelvin \n -> Lambda in gHz \n -> Resolution orders are log2 of number of points desired")
		exit(0)

	#Command line arguments
	omega = float(argv[1])*GHz
	E_J = float(argv[2])*GHz*planck_constant
	nu = float(argv[3])
	Gamma = float(argv[4])*GHz
	gamma_q = float(argv[5])*e_charge**2/THz
	Temp = float(argv[6])*Kelvin
	Lambda = float(argv[7])*GHz
	N_samples = int(argv[8])
	N_periods = int(argv[9])
	N_revivals = int(argv[10])
	N_wells = int(argv[11])
	N_rungs = int(argv[12])
	drive_resolution_order = int(argv[13]) #Number of times in the first window to sample for binary search
	LCJ_save_path = argv[14]
	data_save_path = argv[15]
	
	if len(argv) >= 17:
		seed_SSE = int(argv[16])
		if len(argv) == 18:
			seed_init = int(argv[17])
		else:
			seed_init = int(randbits(31))
	else:
		seed_SSE = int(randbits(31))
		seed_init = int(randbits(31))


	# =============================================================================
	# 1. Parameters
	Z = nu*Klitzing_constant/4	#LC impedance sqrt{L/C}
	L = Z/omega
	C = 1/(Z*omega) 



	# =============================================================================
	# 2. Initialize quantities from parameters

	#Number of outermost well (=python index of central well)
	n0      = N_wells//2
	parity = -1 if n0 % 2 else 1	#Parity factor to correct sign of spin

	# Vector with well indices
	well_vector = arange(-n0,n0+1)

		
	# =============================================================================
	# 3. Initialize bath

	# We want an ohmic bath
	spectral_function = get_J_ohmic(Temp, Lambda,omega0=1)

	### Get bath object
	B = bath(spectral_function,5*Lambda,Lambda/1e5)

	#Compute 1/(hbar*C_R) from Gamma and J(\epsilon_0/\hbar)
	epsilon_0 = sqrt(4*e_charge**2*E_J/C)
	J_0 = spectral_function(epsilon_0/hbar)
	_lambda = (nu*planck_constant*omega/E_J)**0.25/(2*pi)
	jump_op_coeff = sqrt(pi*Gamma/J_0)*_lambda/e_charge
	

	# =============================================================================
	# 4. Initialize system

	### Get system object 
	LCJ_obj = LCJ_circuit(L,C,E_J,(N_wells,N_rungs),mode="0",load_data=1,save_data=1,data_path=LCJ_save_path)

	###  Hamiltonian (as list of matrices, since its block diagonal)
	HL = zeros((N_wells,N_rungs,N_rungs),dtype=complex)

	### Energies and eigenvectors, in case we need it
	VL = zeros((N_wells,N_rungs,N_rungs),dtype=complex)
	EL = zeros((N_wells,N_rungs),dtype=float)

	# List of jump operators (they are also block-diagonal)
	L_list      = zeros((N_wells,N_rungs,N_rungs),dtype=complex)	#Resistor
	LQ_list      = zeros((N_wells,N_rungs,N_rungs),dtype=complex)	#Charge noise

	# Operators to generate displacements due to charge noise in free segment
	xi_a_list  = zeros((N_wells,N_rungs,N_rungs),dtype=complex)
	xi_b_list  = zeros((N_wells,N_rungs,N_rungs),dtype=complex)

	def complex_exponential(phi):
		return exp(-1j*phi) 

	# Construct Hamiltonian and jump operators 
	for n in range(0,N_wells):
		# Get and save Hamiltonian for this well 
		H     = LCJ_obj.H_wl[n]
		HL[n] = 1*H

		# Get componentas of jump operators  in this well 
		X1 = LCJ_obj.get_exp_alpha_phi_operator(1j, n)
		X2 = X1.conj().T
		
		Q  = 2j*e_charge *LCJ_obj.get_d_dphi_operator_w() 
		Phi = LCJ_obj.get_phi_operator_w(n)
		Xarray = array([Q])

		[Ld],[E,V]  =  B.get_ule_jump_operator(Xarray, H,return_ed=True)

		L_list[n] = jump_op_coeff*Ld

		#Jump operator for charge noise
		LQ_list[n] = Q*sqrt(gamma_q)/C
		
		# Save energies in each well 
		EL[n,:]   = E
		
		# Compute generators of charge noise during quarter cycle
		xi_a = Q + flux_quantum *sqrt(C/L)/(2*pi*e_charge)*Phi
		xi_b = Q - flux_quantum *sqrt(C/L)/(2*pi*e_charge)*Phi
		xi_a_list[n]=xi_a
		xi_b_list[n]=xi_b 


	# Get ground state energy of each well
	E0 = EL[:,0]

	# Obtain revival time from the smallest spacing of ground state energies (they are very close to being commensurate!)
	revival_time             = hbar*pi/(2*(-EL[n0,0]+EL[n0+1,0]))

	# Define stabilizer segment time, and driving period
	z_s = 4*ceil(3/(revival_time*Gamma)/4) 
	T_s = revival_time*z_s 


	# =============================================================================
	# 5. Initialize SSE solver 

	### Define inner product, hermitian conjugation, matrix exponentiation and identity operation. 
	# We use lists of matrices to represent operators, to exploit block-diagonal structure

	# Time increment that each instance of sse_evolve evolves over
	time_increment_stab = T_s
	time_increment_free = pi/(2*omega)

	@njit
	def block_multiply_matrix(A,B):
		"""
		Matrix product for block-diagonal matrices A,B
		"""
		#return einsum("abc,acd->abd",A,B)
		out = zeros((A.shape[0],A.shape[1],B.shape[2]),dtype=complex128)
		for i in range(A.shape[0]):
			out[i] = A[i] @ B[i]
		return out
	   

	@njit
	def block_multiply_vector(A,B):
		"""
		Matrix product for block-diagonal matrix A and block vector B 
		"""
		#return einsum("abc,ac->ab",A,B)
		out = zeros((A.shape[0],A.shape[1]),dtype=complex128)
		for i in range(A.shape[0]):
			out[i] = A[i] @ B[i]
		return out
		

	@njit
	def block_hc(mat):
		"""
		hermitian conjugate for block-diagonal matrix
		"""
		return conj(mat.transpose((0,2,1)))

	def block_expm(mat):
		"""
		matrix exponential of block-diagonal matrix
		"""
		out = zeros(mat.shape,dtype=complex128)
		for n in range(0,N_wells):
			out[n] = expm(mat[n])
			
		return out 

	@njit
	def block_expm_herm(mat,alpha):
		"""
		matrix exponential of block-diagonal (hermitian) matrix
		"""
		out = zeros(mat.shape,dtype=complex128)
		for n in range(0,N_wells):
			evals,evecs = eigh(mat[n])
			out[n] = evecs @ diag(exp(alpha*evals).astype(complex128)) @ evecs.conj().T
			
		return out 

	block_norm = LCJ_obj.G.norm
	### Identity operator for block diagonal operators 
	block_id = array([eye(N_rungs,dtype=complex)]*N_wells)

	### Construct SSE solvers for the stabilizer segment
	# Create one for the initialization periods, and another to handle the 
	#	indefinite evolution
	jump_ops_stab = [L_list,LQ_list]
	SSE_solver_stab = sse_evolver(HL,array(jump_ops_stab),time_increment_stab,
		resolution_order=drive_resolution_order,identity=block_id,hc_method=block_hc,
		dot_method_matrix=block_multiply_matrix,dot_method_vector=block_multiply_vector,
		norm_method = block_norm,expm_method=block_expm,seed=seed_SSE)
	SSE_solver_revival = sse_evolver(HL,array(jump_ops_stab),revival_time,
		resolution_order=drive_resolution_order,identity=block_id,hc_method=block_hc,
		dot_method_matrix=block_multiply_matrix,dot_method_vector=block_multiply_vector,
		norm_method = block_norm,expm_method=block_expm,seed=seed_SSE+1)


	
	# =============================================================================
	# 6. Set initial state
	# Potential updates: Start from an HO state?

	psi0 = zeros((N_wells,N_rungs),dtype=complex)

	rng_init = default_rng(seed_init)
	rng_noise = default_rng(seed_SSE+2)

	# Sample bloch vector of qubit uniformly from Bloch sphere
	u , v = rng_init.uniform(low=0.0,high=1.0,size=2)
	theta , phi = arccos(2*u-1) , 2*pi*v 

	# Construct initial state as
	#	cos(theta/2)|0,0,0>> + exp(i*phi)sin(theta/2)|0,0,1>>
	# We assume TWO logical states encoded in the wells with indices congruent to 0,1 mod nu
	for n in range(0,N_wells):
		well_ind = n - n0
		if well_ind % nu == 0:
			psi0[n,0] = cos(theta/2)*exp(-(well_ind)**2*LCJ_obj.sigma**2/(8*LCJ_obj.r**2))
		elif well_ind % nu == 1:
			psi0[n,0] = sin(theta/2)*exp(1j*phi)*exp(-(well_ind)**2*LCJ_obj.sigma**2/(8*LCJ_obj.r**2))
		

	psi0 = psi0/block_norm(psi0)

	# Generate S-gate matrices (used in computing sigma_y)
	S_y_mats = zeros((N_wells,N_rungs,N_rungs),dtype=complex128)
	for n in range(0,N_wells):
		well_ind = n - n0
		H = LCJ_obj.get_H_w(well_ind)
		S_y_mats[n] = expm(-1j*H*revival_time/hbar) 

	#Function to compute the spin expectations
	def spin_expectations(psi):
		#NB! The spin has the wrong sign if n0 % 2 == 1. 
		#	The factor "parity" corrects this
		norm_sq = block_norm(psi)**2

		S_z = parity*LCJ_obj.get_sz_expval(psi)/norm_sq
		
		#Compute <S_x> by applying Hadamard then computing <S_z>
		psi_x = LCJ_obj.apply_quarter_cycle(psi)
		S_x = parity*LCJ_obj.get_sz_expval(psi_x)/norm_sq
		
		#Compute <S_y> by applying S gate, then Hadamard, then computing <S_z>
		psi_y = zeros((N_wells,N_rungs),dtype=complex)
		for n in range(0,N_wells):
			well_ind = n - n0
			psi_y[n,:] = S_y_mats[n] @ psi[n,:]
		psi_y =  LCJ_obj.apply_quarter_cycle(psi_y)
		S_y = parity*LCJ_obj.get_sz_expval(psi_y)/norm_sq
		
		return S_x , S_y, S_z


	# Function to compute the generalized stabilizer 2s
	def generalized_S2s(psi):
		gen_S2s = zeros(N_wells,dtype=complex)
		for i,n in enumerate(well_vector):
			if n == 0:
				gen_S2s[i] = 1.0
				continue
			else:
				gen_S2s[i] = LCJ_obj.get_S2_generalized_expval(psi,n)

		return gen_S2s


	# =============================================================================
	# 7. Solve 


	S1s = zeros((N_samples,N_revivals+1))
	S2s = zeros((N_samples,N_revivals+1))
	gen_S2s = zeros((N_samples,N_revivals+1,N_wells),dtype=complex)
	Jump_record = []
	for sample_num in range(0,N_samples):
		#Time, log of norm of state lost due to truncation after quarter cycles
		t , LN = 0 , 0
		
		# Initialize state 
		psi = array(psi0)

		# Lists of quantum jumps 
		jumplist = zeros((0,2))

		####
		#### Do the initialization periods
		####
		for i in range(N_periods):
		 
			## ****  Stabilizer segment  **** ##
			psi,jumps =  SSE_solver_stab.sse_evolve(psi)

			if len(jumps)>0:    
				jumps[:,0] = jumps[:,0] + t 
				jumplist = concatenate((jumplist,jumps))
	  
			# Move forward in time 
			t += time_increment_stab


			## ****  Free segment  **** ##
			#Construct and apply charge unitary (if necessary)
			if gamma_q > 0:
				a = rng_noise.normal()*sqrt(pi/4+1/2)/sqrt(omega)
				b = rng_noise.normal()*sqrt(pi/4-1/2)/sqrt(omega)
				
				charge_noise_generator = sqrt(gamma_q/(2*C**2))*(a*xi_a_list + b*xi_b_list) 
				charge_noise_unitary = block_expm_herm(charge_noise_generator,-1.0j)
				
				psi = block_multiply_vector(charge_noise_unitary, psi)

			# Evolve via quarter cycle unitary 
			N_old = block_norm(psi)
			psi = LCJ_obj.optimize_gauge(psi)
			psi = LCJ_obj.apply_quarter_cycle(psi)

			# Restore normalization to its pre-quarter cycle value
			psi = psi/block_norm(psi)*N_old
			norm_sq = N_old**2
			
			# Move forward in time by the time it took to go through a quarter cycle
			t = t + pi/(2*omega)

		#end initialization periods loop
		  
				
		norm_sq = block_norm(psi)**2		
		S1s[sample_num,0] = LCJ_obj.get_S1_expval(psi)/norm_sq
		S2s[sample_num,0] = LCJ_obj.get_S2_expval(psi)/norm_sq
		gen_S2s[sample_num,0,:] = generalized_S2s(psi)/norm_sq

		####
		#### Do the indefinite stabilization
		####
		for revival_num in range(0,N_revivals):
			psi,jumps =  SSE_solver_revival.sse_evolve(psi)

			if len(jumps)>0:    
				jumps[:,0] = jumps[:,0] + t 
				jumplist = concatenate((jumplist,jumps))
	  
			# Move forward in time 
			t += time_increment_stab

			# Record stabilizer expectations
			S1s[sample_num,revival_num+1] = LCJ_obj.get_S1_expval(psi)/norm_sq
			S2s[sample_num,revival_num+1] = LCJ_obj.get_S2_expval(psi)/norm_sq
			gen_S2s[sample_num,revival_num+1,:] = generalized_S2s(psi)/norm_sq

		# Record list of jumps
		Jump_record.append(jumplist)



	# =============================================================================
	# 8. Write the data out to disk

	save_file = data_save_path + "data-omega={}GHz-EJ={}GHz-nu={}-Gamma={}GHz-gamma_q={}e^2THz^-1-T={}K.dat".format(argv[1],argv[2],argv[3],argv[4],argv[5],argv[6])
	with open(save_file,'wb') as f:
		#Store simulation params
		params = pack("iiiii",N_samples,N_periods,N_revivals,N_wells,N_rungs)
		f.write(params)

		for i in range(N_samples):
			f.write(pack("d"*(N_revivals+1),*S1s[i]))
			f.write(pack("d"*(N_revivals+1),*S2s[i]))
			for j in range(N_revivals+1):
				for k in range(N_wells):
					f.write(pack("dd",float(real(gen_S2s[i,j,k])),float(imag(gen_S2s[i,j,k]))))
			
			num_jumps = Jump_record[i].shape[0]
			f.write(pack("i",num_jumps))
			if num_jumps > 0:
				for j in range(num_jumps):
					f.write(pack("ff",*Jump_record[i][j])) 
