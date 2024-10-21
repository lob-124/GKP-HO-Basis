from basic             import *
from units             import *
from numpy             import *
from numpy.linalg      import *
from scipy.integrate   import quad
from scipy.interpolate import interp1d
from matplotlib.pyplot import *

from numpy.random         import default_rng 
from gaussian_bath        import bath, get_J_ohmic
from LCJ_circuit          import LCJ_circuit
from generate_time_series import Time_Series 
from SSE_evolver          import sse_evolver 

from numba import njit

from struct import pack,unpack
from secrets import randbits

from time import perf_counter

"""
Implements 1/f noise during stabilizer segments by multiplying by exp(-i alpha Phi/L*hbar), where
		alpha = int_t1^t2 xi(t) dt is the integral of the noise.   
	Dynamic Decoupling is now done via the Uhrig Pulse Sequence, i.e. we implement UDD

Author: Liam O'Brien
"""


if __name__ == "__main__":
	from sys import argv

	if len(argv) not in [19,20,21]:
		print("Usage: omega nu E_J/h gamma_phi Gamma T Lambda N_decoupling N_samples N_periods N_wells N_rungs <drive_resolution_order_seg1> <drive_resolution_order_seg2> <coeffs_file> <noise_file> <LCJ_save_path> <data_save_path> seed_SSE (op) seed_init (op)")
		print("Units: \n -> omega in GHz \n -> E_J in GHz \n -> gamma_q in e^2/THz \n -> gamma_phi in Phi_0^2/THz \n -> gamma in GHz \n -> gamma_q in e^2/THz \n -> gamma_pL in kHz \n -> Temp in Kelvin \n -> Lambda in gHz  \n -> Resolution orders are log2 of number of points desired")
		print("gamma_phi is PSD magnitude at (regular) frequency f = 1 Hz")
		print("N_decoupling is the number of dynamic decoupling steps. This splits the phi^4 step of the gate protocol into N_decoupling + 1 equal-length steps, with a free evolution for pi/omega in between")
		exit(0)

	#Command line arguments
	omega = float(argv[1])*GHz
	nu = float(argv[2])
	E_J = float(argv[3])*GHz*planck_constant
	gamma_phi = float(argv[4])*flux_quantum**2/THz
	Gamma = float(argv[5])*GHz
	Temp = float(argv[6])*Kelvin
	Lambda = float(argv[7])*GHz
	N_decoupling = int(argv[8])
	N_samples = int(argv[9])
	N_periods = int(argv[10])
	N_wells = int(argv[11])
	N_rungs = int(argv[12])
	drive_resolution_order_seg1 = int(argv[13]) #Number of times in the first window to sample for binary search
	drive_resolution_order_seg2 = int(argv[14])
	coeffs_file = argv[15]
	noise_file = argv[16]
	LCJ_save_path = argv[17]
	data_save_path = argv[18]

	if len(argv) >= 20:
		seed_SSE = int(argv[19])
		if len(argv) == 21:
			seed_init = int(argv[20])
		else:
			seed_init = int(randbits(31))
	else:
		seed_SSE = int(randbits(31))
		seed_init = int(randbits(31))

	#Extract the coefficients of the "+" potential from the extra circuit elements
	with open(coeffs_file,'r') as f:
		_coeffs = array([float(line.rstrip("e \n")) for line in f])*hbar*GHz
		
	#Extract the sign of C_4. This determines the sign of the phase imparted between |+z> and |-z>.
	phase_sign = 1 if _coeffs[2] > 0 else -1


	#Read in the time series for the noise
	with open(noise_file,'rb') as f:
		num_realizations,num_t_points = unpack("ii",f.read(8))

		noise_data = zeros((num_realizations,num_t_points))
		for i in range(num_realizations):
			noise_data[i] = sqrt(gamma_phi)*array(unpack("f"*num_t_points,f.read(4*num_t_points)))

	#Create interpolation functions for the noise signals (used to integrate the signal)
	t_points = linspace(0,1e-4,num=num_t_points)*second	    #NB: endpoints of 0,100 microsec hard-coded. Change in future?
	noise_signals = [interp1d(t_points,noise_sig,kind='cubic') for noise_sig in noise_data]



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

	
	# =============================================================================
	# 3. Initialize bath

	# We want an ohmic bath for the resistor
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
	H_LCJ_list = zeros((N_wells,N_rungs,N_rungs),dtype=complex)
	H_LCJ_plus_list = zeros((N_wells,N_rungs,N_rungs),dtype=complex)

	### Energies of the LCJ and segment 1 well Hamiltonians
	energies_LCJ = zeros((N_wells,N_rungs),dtype=float)
	energies_LCJ_plus = zeros((N_wells,N_rungs),dtype=float)

	# List of jump operators (they are also block-diagonal)
	L_LCJ_list      = zeros((N_wells,N_rungs,N_rungs),dtype=complex)	#Resistor for LCJ circuit
	L_LCJ_plus_list     = zeros((N_wells,N_rungs,N_rungs),dtype=complex)	#Resistor for LCJ+ circuit 


	#Get charge operator (same in every well)
	Q  = 2j*e_charge *LCJ_obj.get_d_dphi_operator_w()
	Xarray = array([Q])

	#Charge and flux operators in the grid basis (used for generating noise operators)
	Q_list = repeat(array([Q]),N_wells,axis=0)
	Flux_list = zeros((N_wells,N_rungs,N_rungs),dtype=complex)


	# Construct Hamiltonian and jump operators 
	for n in range(0,N_wells):
		# Get and save LCJ Hamiltonian for this well 
		_H_LCJ     = LCJ_obj.H_wl[n]
		H_LCJ_list[n] = 1*_H_LCJ

		#Get and save the LCJ+ Hamiltonian for this well
		H_LCJ_plus_list[n] = 1*_H_LCJ
		_Phi = LCJ_obj.get_phi_operator_w(n-n0,dim=N_rungs+2*(len(_coeffs)-1))
		_Phi2 = _Phi @ _Phi
		_curr = _Phi2

		#Add in the "+" terms from the extra circuit elements (as power series in Phi)
		for coeff in _coeffs[1:]:	#NB: we skip the 0th order term (it's just a global phase)
			H_LCJ_plus_list[n] += coeff * _curr[:N_rungs,:N_rungs]
			_curr = _curr @ _Phi2


		#Save flux operator in this well (for flux noise)
		Flux_list[n] = _Phi[:N_rungs,:N_rungs]*flux_quantum/(2*pi)
		

		# Get the resistor jump operators
		#  Note there are two of them now, one for each gate segment
		# First, the LCJ jump oeprators (phi^2 and stabilizer segments)
		[Ld],[E,V]  =  B.get_ule_jump_operator(Xarray,_H_LCJ,return_ed=True)
		L_LCJ_list[n] = jump_op_coeff*Ld
		energies_LCJ[n,:]   = array(E)

		# Seconds, the LCJ+ jump operators (phi^4 segment)
		[Ld2],[E2,V2]  =  B.get_ule_jump_operator(Xarray,H_LCJ_plus_list[n],return_ed=True)
		L_LCJ_plus_list[n] = jump_op_coeff*Ld2
		energies_LCJ_plus[n,:]   = array(E2)

	
	# Obtain revival time from the smallest spacing of ground state energies (they are very close to being commensurate!)
	revival_time             = hbar*pi/(2*(-energies_LCJ[n0,0]+energies_LCJ[n0+1,0]))

	# Define stabilizer segment time, and driving period
	z_s = 4*ceil(3/(revival_time*Gamma)/4)
	T_stab = revival_time*z_s 
	driving_period = T_stab + pi/(2*omega)


	#Fit the energies to a polynomial in well index N
	fit_max = min(int(ceil((6/_lambda)/(2*pi))),N_wells//2)
	params_E = polyfit(list(range(-fit_max,fit_max+1)),energies_LCJ_plus[n0-fit_max:n0+fit_max+1,0],8)


	#Define the phi^4 segment time t_phi4 and phi^2 segment time t_phi2 (based on the fitted values above)
	t_phi4 = abs(pi*hbar/(8*params_E[-5]))		#Time to align phi^4 phases
	t_revival_phi4 = abs(pi*hbar/(2*(params_E[-3])))
	num_revivals = t_phi4/t_revival_phi4	#Number of revivals (from quadratic terms) that have elapsed in this time
	t_phi2 = (4 - (num_revivals % 4))*revival_time	#Amount of extra (LCJ!) time needed for all phi^2 phases to align to 0 (mod 2pi)
	
	gate_time = t_phi4 + t_phi2	#total time to do the gate (without cleanup steps)
	total_time = gate_time + N_periods*driving_period	#total simulation time (including cleanup steps)


	# =============================================================================
	# 5. Initialize SSE integrator 

	### Define inner product, hermitian conjugation, matrix exponentiation and identity operation. 
	# We use lists of matrices to represent operators, to exploit block-diagonal structure

	# Time increment that each instance of sse_evolve evolves over
	time_increments_phi4 = ([t_phi4*(sin(pi*(j+1)/(2*(N_decoupling+1)))**2 - 
							sin(pi*j/(2*(N_decoupling+1)))**2) for j in range(0,N_decoupling+1)])
	time_increment_phi2 = t_phi2
	time_increment_stab = T_stab


	@njit
	def block_multiply_matrix(A,B):
		"""
		Matrix product for block-diagonal matrices A,B
		"""
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

	#Norm and Identity for block-diagonal objects
	block_norm = LCJ_obj.G.norm
	block_id = array([eye(N_rungs,dtype=complex)]*N_wells)


	### ***
	###    Construct SSE solvers
	### ***
	#First, for the fixed-length parts  
	#  1: for the phi^4 gate segment
	#      We generate a list of them, since the sequences in UDD are not necessarily equal
	jump_ops_LCJ_plus = [L_LCJ_plus_list]
	SSE_solvers_seg1 = []
	for i in range(0,N_decoupling+1):
		_SSE_solver = sse_evolver(H_LCJ_plus_list,array(jump_ops_LCJ_plus), time_increments_phi4[i],
			resolution_order=drive_resolution_order_seg1,identity=block_id,hc_method=block_hc,
			dot_method_matrix=block_multiply_matrix,dot_method_vector=block_multiply_vector,
			norm_method = block_norm,expm_method=block_expm,seed=seed_SSE)
		SSE_solvers_seg1.append(_SSE_solver)
	
	#  2: for the phi^2 gate segment
	jump_ops_LCJ = [L_LCJ_list]
	SSE_solver_seg2 = sse_evolver(H_LCJ_list,array(jump_ops_LCJ), time_increment_phi2,
		resolution_order=drive_resolution_order_seg2,identity=block_id,hc_method=block_hc,
		dot_method_matrix=block_multiply_matrix,dot_method_vector=block_multiply_vector,
		norm_method = block_norm,expm_method=block_expm,seed=seed_SSE+1)

	#  3: for the subsequent stabilizer segments
	#	 Note we need this because the time interval is different than in the phi^2 segment
	SSE_solver_stab = sse_evolver(H_LCJ_list,array(jump_ops_LCJ), time_increment_stab,
		resolution_order=drive_resolution_order_seg2,identity=block_id,hc_method=block_hc,
		dot_method_matrix=block_multiply_matrix,dot_method_vector=block_multiply_vector,
		norm_method = block_norm,expm_method=block_expm,seed=seed_SSE+2)



	# =============================================================================
	# 6. Define functions to generate noise evolution operators during free segment 

	#The unitary capturing evolution due to the noise operators (in the co-rotating frame of the
	#	LC Hamiltonian) is given by:
	#
	#					exp[-i(A(t_1,t_2)q + B(t_1,t_2)phi)/hbar]
	#
	#	where A(t_1,t_2) and B(t_1,t_2) involve integrals of the noise signal, and are implemented below
	
	def free_segment_exp_coeffs(xi_phi,t_1,t_2):
		_f1 = lambda t: sin(omega*t)*xi_phi(t)
		_f2 = lambda t: cos(omega*t)*xi_phi(t)
		return (flux_quantum*nu/(2*e_charge*L))*quad(_f1,t_1,t_2)[0] , (1/L)*quad(_f2,t_1,t_2)[0]



	
	# =============================================================================
	# 7. Set initial state

	psi0 = zeros((N_wells,N_rungs),dtype=complex)

	rng = default_rng(seed_init)

	# Set bloch angles of qubit
	theta_0 , phi_0 = pi/2 , 0		#Start in a |+x> state 
	
	# Construct initial state as
	#	cos(theta/2)|0,0,0>> + exp(i*phi)sin(theta/2)|0,0,1>>
	# We assume TWO logical states encoded in the wells with indices congruent to 0,1 mod nu
	for n in range(0,N_wells):
		well_ind = n - n0
		if well_ind % nu == 0:
			psi0[n,0] = cos(theta_0/2)*exp(-(well_ind)**2*LCJ_obj.sigma**2/(8*LCJ_obj.r**2))
		elif well_ind % nu == 1:
			psi0[n,0] = sin(theta_0/2)*exp(1j*phi_0)*exp(-(well_ind)**2*LCJ_obj.sigma**2/(8*LCJ_obj.r**2))

	psi0 = psi0/block_norm(psi0)
	print("Initialized!")

	

	S_y_mats = zeros((N_wells,N_rungs,N_rungs),dtype=complex128)
	for n in range(0,N_wells):
		S_y_mats[n] = expm(-1j*H_LCJ_list[n]*revival_time/hbar) 

	#Function to compute the spin expectations
	def spin_expectations(psi):
		#NB! The spin has the wrong sign if n0 % 2 == 1. Not sure why yet...
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


	#Function to compute the overlap <phi|psi> where |phi> is the "target" state.
	#	Does so using the Bloch sphere paramterization of the state as input 
	def overlap(theta,phi,theta_target,phi_target):
		return cos(theta/2)*cos(theta_target/2) + sin(theta/2)*sin(theta_target/2)*exp(1j*(phi-phi_target))


	#Function implementing evolution by half an LC period. This has the effect of sending phi -> -phi
	signs = (-1)**array(range(N_rungs))		#array of signs (-1)^nu for nu = 0,...,N_rungs-1
	def half_period_LC_evolution(psi):
		res = zeros((N_wells,N_rungs),dtype=complex128)

		#For a wavefunction 
		#			|psi> = sum_{N,nu} C^N_nu |N,nu>
		#	in the grid basis, the half-period evolution has the effect of sending
		#			C^N_nu -> (-1)^nu*C^{-N}_nu
		for n in range(N_wells):
			res[n] = signs*psi[N_wells-1-n]

		return res


	# =============================================================================
	# 8. Solve 
	
	S1s = zeros((N_samples))
	S2s = zeros((N_samples))
	spins  = zeros((N_samples,3)) 
	fidelities = zeros(N_samples)
	FT_weights = zeros((N_samples,N_periods))
	Jump_record = []

	
	for sample_num in range(0,N_samples):
		#Time, log of norm of state lost due to truncation after quarter cycles
		t , LN = 0 , 0
		
		# Initialize state 
		psi = array(psi0)

		# Lists of quantum jumps 
		jumplist = zeros((0,2))


		####
		#### Do the T gate evolution
		####
		## First, the phi^4 segment (with dynamical decoupling)
		for n in range(N_decoupling+1):
			
			# Evolve between the half-cycle rotations
			psi,jumps =  SSE_solvers_seg1[n].sse_evolve(psi)
			if len(jumps)>0:    
				jumps[:,0] = jumps[:,0] + t 
				jumplist = concatenate((jumplist,jumps))

			# Apply the (random) phase operator from the noise
			alpha , _ = quad(noise_signals[0],t,t+time_increments_phi4[n])
			noise_unitary_stab = block_expm(-1j*alpha/(hbar*L)*Flux_list)#block_expm_herm(Flux_list,-1j*alpha/(hbar*L))
			psi = block_multiply_vector(noise_unitary_stab,psi)

			t += time_increments_phi4[n]

			#Do free evolution for half of an LC period. This sends phi -> -phi, and implements dynamical
			#	decoupling
			if n < N_decoupling:
				# Apply charge and flux noise, in a similar manner as with the quarter cycle
				c_q , c_phi = free_segment_exp_coeffs(noise_signals[0],t,t+pi/omega)
				noise_unitary_free = block_expm_herm((c_q*Q_list + c_phi*Flux_list),-1j/hbar)
				psi = block_multiply_vector(noise_unitary_free,psi)
				
				# And apply the half-period evolution
				psi = half_period_LC_evolution(psi)

				t += pi/omega
				

		###
		###    Now do the phi^2 segment
		###
		psi,jumps =  SSE_solver_seg2.sse_evolve(psi)
		if len(jumps)>0:    
			jumps[:,0] = jumps[:,0] + t 
			jumplist = concatenate((jumplist,jumps))

		# Apply the (random) phase operator from the noise
		alpha , _ = quad(noise_signals[0],t,t+time_increment_phi2)
		noise_unitary_stab = block_expm_herm(Flux_list,-1j*alpha/(hbar*L))
		psi = block_multiply_vector(noise_unitary_stab,psi)
			
		# Move forward in time 
		t += time_increment_phi2


		###
		### Now do the normal protocol for a few cycles
		###  
		for nperiod in range(N_periods):
		
		## ****  Stabilizer segment  **** ##
			psi,jumps =  SSE_solver_stab.sse_evolve(psi)
			if len(jumps)>0:    
				jumps[:,0] = jumps[:,0] + t 
				jumplist = concatenate((jumplist,jumps))

			# Apply the (random) phase operator from the noise
			alpha , _ = quad(noise_signals[0],t,t+time_increment_stab)
			noise_unitary_stab = block_expm_herm(Flux_list,-1j*alpha/(hbar*L))
			psi = block_multiply_vector(noise_unitary_stab,psi)

			# Move forward in time 
			t += time_increment_stab


		## ****  Free segment  **** ##

			#Generate (unitary) charge/flux noise evolution
			c_q , c_phi = free_segment_exp_coeffs(noise_signals[0],t,t+pi/(2*omega))
			noise_unitary_free = block_expm(-1j*(c_q*Q_list + c_phi*Flux_list)/hbar)

			#Apply noise unitary
			psi = block_multiply_vector(noise_unitary_free,psi)

			# Evolve via quarter cycle unitary 
			N_old = block_norm(psi)
			psi = LCJ_obj.optimize_gauge(psi)
			psi = LCJ_obj.apply_quarter_cycle(psi)

			# Record how much weight is lost due to Hilbert space truncation after applying quarter
			#	cycle unitary
			LN += log(block_norm(psi)**2/N_old**2)
			FT_weights[sample_num,nperiod] = exp(LN)
			
			# Restore normalization to its pre-quarter cycle value
			psi = psi/block_norm(psi)*N_old
			norm_sq = N_old**2
				
			# Move forward in time 
			t += pi/(2*omega)
		

		#Calculate quantities of interest for this trajectory 
		norm_sq = block_norm(psi)**2
		S1s[sample_num]   = LCJ_obj.get_S1_expval(psi)/norm_sq  
		S2s[sample_num]   = LCJ_obj.get_S2_expval(psi)/norm_sq
		spins[sample_num] = spin_expectations(psi)
		S_x,S_y,S_z = spins[sample_num]
		theta, phi = arccos(S_z), arctan2(S_y,S_x)
		fidelities[sample_num] = abs(overlap(theta,phi,theta_0,phi_0-phase_sign*pi/8))**2
		Jump_record.append(jumplist)

	#end loop over SSE trajectories


	# =============================================================================
	# 9. Write the data out to disk

	save_file = data_save_path + "data-omega={}GHz-nu={}-EJ={}GHz-gamma_phi={}Phi_0^2THz^-1-N_decoup={}-gamma={}GHz-T={}K.dat".format(argv[1],argv[2],argv[3],argv[4],argv[8],argv[5],argv[6])
	with open(save_file,'wb') as f:
		#Store simulation params
		params = pack("iiii",N_samples,N_periods,N_wells,N_rungs)
		f.write(params)
		f.write(pack("f",gate_time))

		for i in range(N_samples):
			f.write(pack("f",S1s[i]))
			f.write(pack("f",S2s[i]))
			f.write(pack("f"*3,*spins[i]))
			
			num_jumps = Jump_record[i].shape[0]
			f.write(pack("i",num_jumps))
			if num_jumps > 0:
				for j in range(num_jumps):
					f.write(pack("ff",*Jump_record[i][j])) 

			f.write(pack("d",fidelities[i]))
			f.write(pack("d"*N_periods,*FT_weights[i]))
	