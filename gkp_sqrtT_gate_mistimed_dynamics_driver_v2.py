from basic             import *
from units             import *
from numpy             import *
from numpy.linalg      import *
from matplotlib.pyplot import *

from numpy.random      import default_rng 
from gaussian_bath     import bath, get_J_ohmic
from LCJ_circuit       import LCJ_circuit
from SSE_evolver       import sse_evolver

from time import perf_counter
from numba import njit

from struct import pack
from secrets import randbits

"""
v2: A new version of this driver that chooses mistimings of every segment randomly 
	in the interval [-delta_t,delta_t]

Author: Liam O'Brien
"""


if __name__ == "__main__":
	from sys import argv

	if len(argv) not in [18,19,20]:
		print("Usage: omega nu E_J/h delta_t Gamma T Lambda N_samples N_periods N_wells N_rungs <drive_resolution_order_seg1> <drive_resolution_order_seg2> <drive_resolution_mistiming> <coeffs_file> <LCJ_save_path> <data_save_path> seed_SSE (op) seed_init (op)")
		print("Units: \n -> omega in GHz \n -> E_J in GHz \n -> delta_t in ps \n -> gamma in GHz \n -> gamma_q in e^2/THz \n -> gamma_pL in kHz \n -> Temp in Kelvin \n -> Lambda in gHz \n -> Resolution orders are log2 of number of points desired")
		print("Note that 2*delta_t is width of mistiming distribution (uniform about 0) of mistiming for all switches")
		exit(0)

	#Command line arguments
	omega = float(argv[1])*GHz
	nu = float(argv[2])
	E_J = float(argv[3])*GHz*planck_constant
	delta_t = float(argv[4])*picosecond
	Gamma = float(argv[5])*GHz
	Temp = float(argv[6])*Kelvin
	Lambda = float(argv[7])*GHz
	N_samples = int(argv[8])
	N_periods = int(argv[9])
	N_wells = int(argv[10])
	N_rungs = int(argv[11])
	drive_resolution_order_seg1 = int(argv[12]) #Number of times in the first window to sample for binary search
	drive_resolution_order_seg2 = int(argv[13])
	drive_resolution_mistiming  = int(argv[14])		#log2(resolution) of mistiming segments
	coeffs_file = argv[15]
	LCJ_save_path = argv[16]
	data_save_path = argv[17]

	#Seeds for the SSE solvers
	if len(argv) >= 19:
	    seed_SSE = int(argv[18])
	    if len(argv) == 20:
	        seed_init = int(argv[19])
	    else:
	    	seed_init = int(randbits(31))
	else:
		seed_SSE = int(randbits(31))
		seed_init = int(randbits(31))

	#Read in the coefficients for the "+" potential from the extra circuit elements
	with open(coeffs_file,'r') as f:
		_coeffs = array([float(line.rstrip("e \n")) for line in f])*hbar*GHz

	#Capture sign of phi^4 coefficient to make sure we get sign of expected phase correct
	phase_sign = 1 if _coeffs[2] > 0 else -1	 


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

	# Generate spectral function for an ohmic bath
	spectral_function = get_J_ohmic(Temp, Lambda,omega0=1)

	# Generate bath object
	B = bath(spectral_function,5*Lambda,Lambda/1e5)

	#Compute 1/(hbar*C_R) from Gamma and J(\epsilon_0/\hbar)
	epsilon_0 = sqrt(4*e_charge**2*E_J/C)
	J_0 = spectral_function(epsilon_0/hbar)
	_lambda = (nu*planck_constant*omega/E_J)**0.25/(2*pi)
	jump_op_coeff = sqrt(pi*Gamma/J_0)*_lambda/e_charge



	# =============================================================================
	# 4. Initialize system

	# Get system object 
	LCJ_obj = LCJ_circuit(L,C,E_J,(N_wells,N_rungs),mode="0",load_data=1,save_data=1,data_path=LCJ_save_path)

	# Hamiltonian (as list of matrices, since its block diagonal)
	H_LC   = zeros((N_wells,N_rungs,N_rungs),dtype=complex)
	H_LCJ  = zeros((N_wells,N_rungs,N_rungs),dtype=complex)
	H_LCJ_plus = zeros((N_wells,N_rungs,N_rungs),dtype=complex)

	# Energies of the LCJ and LCJ+ well Hamiltonians
	energies_LCJ = zeros((N_wells,N_rungs),dtype=float)
	energies_LCJ_plus = zeros((N_wells,N_rungs),dtype=float)

	# List of jump operators (they are also block-diagonal)
	L_LC_list       = zeros((N_wells,N_rungs,N_rungs),dtype=complex)    #Resistor for LC circuit*
	L_LCJ_list      = zeros((N_wells,N_rungs,N_rungs),dtype=complex)	#Resistor for LCJ circuit
	L_LCJ_plus_list     = zeros((N_wells,N_rungs,N_rungs),dtype=complex)	#Resistor for LCJ+ circuit


	# *Not technically present, but needed for SSE_solver() to work properly. We model this is a dramatic suppression
	#	of the resistor coupling 


	#Charge operator (same in every well)
	Q  = 2j*e_charge *LCJ_obj.get_d_dphi_operator_w() 
	Xarray = array([Q]) 	#operator appearing in system-bath coupling (Q for capacitive coupling)

	# Construct Hamiltonian and jump operators in each well
	for n in range(0,N_wells):
	    #Get and save the LC Hamitlonian for this well
	    _H_LC = LCJ_obj.compute_H_LC_w(n-n0,dim = "sample_numgs")
	    H_LC[n] = array(_H_LC)


	    # Get and save the LCJ Hamiltonian for this well 
	    _H_LCJ     = LCJ_obj.H_wl[n]
	    H_LCJ[n] = array(_H_LCJ)

	    # Compute and save LCJ+ Hamiltonian for this well
	    H_LCJ_plus[n] = array(_H_LCJ)
	    _Phi = LCJ_obj.get_phi_operator_w(n-n0,dim=N_rungs+2*(len(_coeffs)-1))
	    _Phi2 = _Phi @ _Phi
	    _curr = _Phi2

	    #Add in "+" terms read in from file above
	    for coeff in _coeffs[1:]:
	    	H_LCJ_plus[n] += coeff * _curr[:N_rungs,:N_rungs]
	    	_curr = _curr @ _Phi2	#Our circuit only generates even powers of phi


	    # Compute and save the resistor jump operators in this well
	    #  Note there are three of them now: for the LC circuit, for the LCJ circuit, and for 
	    #		the LCJ+ circuit
	    [Ld_LC],[E_LC,V_LC]  =  B.get_ule_jump_operator(Xarray,_H_LC,return_ed=True)
	    L_LC_list[n] = 1e-10*jump_op_coeff*Ld_LC

	    [Ld],[E,V]  =  B.get_ule_jump_operator(Xarray,_H_LCJ,return_ed=True)
	    L_LCJ_list[n] = jump_op_coeff*Ld
	    energies_LCJ[n,:]   = array(E)

	    [Ld2],[E2,V2]  =  B.get_ule_jump_operator(Xarray,H_LCJ_plus[n],return_ed=True)
	    L_LCJ_plus_list[n] = jump_op_coeff*Ld2
	    energies_LCJ_plus[n,:]   = array(E2)


	# Obtain revival time from the smallest spacing of ground state energies (they are very close to being commensurate!)
	revival_time             = hbar*pi/(2*(-energies_LCJ[n0,0]+energies_LCJ[n0+1,0]))

	# Define stabilizer segment time
	z_s = 4*ceil(3/(revival_time*Gamma)/4) 
	T_stab = revival_time*z_s 


	#Fit the LCJ+ energies to a polynomial in well index N
	fit_max = int(ceil((6/_lambda)/(2*pi)))
	params_E = polyfit(list(range(-fit_max,fit_max+1)),energies_LCJ_plus[n0-fit_max:n0+fit_max+1,0],8)
	

	#Compute the gate times (based on the fitted values above)
	t_phi4 = abs(pi*hbar/(8*params_E[-5]))		        #Time to align phi^4 phases
	t_revival_phi4 = abs(pi*hbar/(2*(params_E[-3])))	#Time to align phi^2 under evo by LCJ+ Ham
	num_revivals = t_phi4/t_revival_phi4		        #Number of phi^2 revivals that have elapsed in this time
	t_phi2 = (4 - (num_revivals % 4))*revival_time	    #Amount of extra (LCJ!) time needed for phi^2 phases to align
	
	gate_time = t_phi4 + t_phi2 	#total gate time (without cleanup steps)

	
	# =============================================================================
	# 5. Initialize SSE solvers 

	# Time increment that each instance of sse_evolve evolves over
	#  For each segment with the resistor, we modify the evolution time T -> T - delta_t, and 
	#   then draw the mistiming uniformly on [0,2*delta_t] (explained further below)
	time_increment_phi4 = t_phi4 - delta_t
	time_increment_phi2 = t_phi2 - delta_t
	time_increment_stab = T_stab - delta_t


	# Define inner product, hermitian conjugation, matrix exponentiation and identity operation. 
	# We use lists of matrices to represent operators, to exploit block-diagonal structure

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
	    out = zeros((A.shape[0],A.shape[1]),dtype=complex128)
	    for i in range(A.shape[0]):
	        out[i] = A[i] @ B[i]
	    return out
	    

	@njit
	def block_hc(mat):
	    """
	    Hermitian conjugate for block-diagonal matrix
	    """
	    return conj(mat.transpose((0,2,1)))


	def block_expm(mat):
	    """
	    Matrix exponential of block-diagonal matrix
	    """
	    out = zeros(mat.shape,dtype=complex128)
	    for n in range(0,N_wells):
	        out[n] = expm(mat[n])
	        
	    return out 


	@njit
	def block_expm_herm(mat,alpha):
	    """
	    Matrix exponential of block-diagonal (hermitian) matrix

	    NB: diagonalization is faster than expm() for hermitian matrices with jit'ing
	    """
	    out = zeros(mat.shape,dtype=complex128)
	    for n in range(0,N_wells):
	        evals,evecs = eigh(mat[n])
	        out[n] = evecs @ diag(exp(alpha*evals).astype(complex128)) @ evecs.conj().T
	        
	    return out 

	# Norm and identity operator for block-diagonal objects
	block_norm = LCJ_obj.G.norm
	block_id = array([eye(N_rungs,dtype=complex)]*N_wells)


	### ***
	###    Construct SSE solvers
	### ***

	#First, for the fixed-length parts  
	#  1: for the phi^4 gate segment
	jump_ops_LCJ_plus = [L_LCJ_plus_list]
	SSE_solver_seg1 = sse_evolver(H_LCJ_plus,array(jump_ops_LCJ_plus), time_increment_phi4,
		resolution_order=drive_resolution_order_seg1,identity=block_id,hc_method=block_hc,
		dot_method_matrix=block_multiply_matrix,dot_method_vector=block_multiply_vector,
		norm_method = block_norm,expm_method=block_expm,seed=seed_SSE)
	
	#  2: for the phi^2 gate segment
	jump_ops_LCJ = [L_LCJ_list]
	SSE_solver_seg2 = sse_evolver(H_LCJ,array(jump_ops_LCJ), time_increment_phi2,
		resolution_order=drive_resolution_order_seg2,identity=block_id,hc_method=block_hc,
		dot_method_matrix=block_multiply_matrix,dot_method_vector=block_multiply_vector,
		norm_method = block_norm,expm_method=block_expm,seed=seed_SSE+1)

	#  3: for the subsequent stabilizer segments
	#	 Note we need this because the time interval is different than in the phi^2 segment
	SSE_solver_stab = sse_evolver(H_LCJ,array(jump_ops_LCJ), time_increment_stab,
		resolution_order=drive_resolution_order_seg2,identity=block_id,hc_method=block_hc,
		dot_method_matrix=block_multiply_matrix,dot_method_vector=block_multiply_vector,
		norm_method = block_norm,expm_method=block_expm,seed=seed_SSE+2)


	#Second, for the mistiming in each part
	#	For all segments with the resistor, we draw mistimings from [0,2*delta_t], and reduce the 
	#	 "normal" evolution by delta_t. This is because we can't evolve backwards when there are 
	#	 quantum jumps (as these are not reversible).

	#  1: for the phi^4 segment
	SSE_solvers_LCJ_plus_mistiming = []
	for i in range(drive_resolution_mistiming+1):
		SSE_solvers_LCJ_plus_mistiming.append(sse_evolver(H_LCJ_plus,array(jump_ops_LCJ_plus), delta_t/2**i,
			resolution_order=(drive_resolution_mistiming-i),identity=block_id,hc_method=block_hc,
			dot_method_matrix=block_multiply_matrix,dot_method_vector=block_multiply_vector,
			norm_method = block_norm,expm_method=block_expm,seed=seed_SSE+3+i))
		
	#  2: for the phi^2 segment and stabilizer segments
	#	Note that we can use the same SSE solvers for mistiming in these two segments, since the 
	#	 Hamiltonian is the same
	SSE_solvers_LCJ_mistiming = []
	for i in range(drive_resolution_mistiming+1):
		SSE_solvers_LCJ_mistiming.append(sse_evolver(H_LCJ,array(jump_ops_LCJ), delta_t/2**i,
			resolution_order=(drive_resolution_mistiming-i),identity=block_id,hc_method=block_hc,
			dot_method_matrix=block_multiply_matrix,dot_method_vector=block_multiply_vector,
			norm_method = block_norm,expm_method=block_expm,seed=seed_SSE+3+drive_resolution_mistiming+i))
		
	#  3: for the mistimed free segments
	#	 For the free segment, we draw from [-delta_t,delta_t]
	#	 There are no jumps during the free segment, so evolving backwards is not an issue
	jump_ops_LC = [L_LC_list]
	SSE_solvers_LC_mistiming_pos = []
	SSE_solvers_LC_mistiming_neg = []
	for i in range(drive_resolution_mistiming):
		#Create separate solvers for evolution forwards and backwards in time
		SSE_solvers_LC_mistiming_pos.append(sse_evolver(H_LC,array(jump_ops_LC), delta_t/2**(i+1),
			resolution_order=(drive_resolution_mistiming-i-1),identity=block_id,hc_method=block_hc,
			dot_method_matrix=block_multiply_matrix,dot_method_vector=block_multiply_vector,
			norm_method = block_norm,expm_method=block_expm,seed=seed_SSE+3+2*drive_resolution_mistiming+i))
		SSE_solvers_LC_mistiming_neg.append(sse_evolver(H_LC,array(jump_ops_LC),-delta_t/2**(i+1),
			resolution_order=(drive_resolution_mistiming-i-1),identity=block_id,hc_method=block_hc,
			dot_method_matrix=block_multiply_matrix,dot_method_vector=block_multiply_vector,
			norm_method = block_norm,expm_method=block_expm,seed=seed_SSE+3+3*drive_resolution_mistiming+i))




	# =============================================================================
	# 6. Set initial state

	rng = default_rng(seed_init)

	# Set bloch sphere angles of qubit
	theta_0 , phi_0 = pi/2 , 0		#Start in a |+x> state 
	
	# Construct initial state as
	#	cos(theta/2)|0,0,0>> + exp(i*phi)sin(theta/2)|0,0,1>>
	# We assume TWO logical states encoded in the wells with indices congruent to 0,1 mod nu
	psi0 = zeros((N_wells,N_rungs),dtype=complex)
	for n in range(0,N_wells):
	    well_ind = n - n0
	    if well_ind % nu == 0:
	        psi0[n,0] = cos(theta_0/2)*exp(-(well_ind)**2*LCJ_obj.sigma**2/(8*LCJ_obj.r**2))
	    elif well_ind % nu == 1:
	        psi0[n,0] = sin(theta_0/2)*exp(1j*phi_0)*exp(-(well_ind)**2*LCJ_obj.sigma**2/(8*LCJ_obj.r**2))

	psi0 = psi0/block_norm(psi0)
	
	# Generate S-gate matrices (used in computing sigma_y)
	S_y_mats = zeros((N_wells,N_rungs,N_rungs),dtype=complex128)
	for n in range(0,N_wells):
	    S_y_mats[n] = expm(-1j*H_LCJ[n]*revival_time/hbar) 

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


	#Function to compute the overlap <phi|psi> where |phi> is the "target" state.
	#	Does so using the Bloch sphere angles of the state as input 
	def overlap(theta,phi,theta_target,phi_target):
		return cos(theta/2)*cos(theta_target/2) + sin(theta/2)*sin(theta_target/2)*exp(1j*(phi-phi_target))




	# =============================================================================
	# 7. Solve 
	
	#Arrays to store output data in
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
	    #### Do the phi^4 segment
	    ####
	    psi,jumps =  SSE_solver_seg1.sse_evolve(psi)
	    norm_sq = block_norm(psi)**2

	    if len(jumps)>0:    
	        jumps[:,0] = jumps[:,0] + t 
	        jumplist = concatenate((jumplist,jumps))

	            
        # Move forward in time 
	    t+=time_increment_phi4
	    
	    #Do the phi^4 mistiming
	    mistiming_phi4 = 2*delta_t*rng.random() #draw from [0,2*delta_t]
	    dt = 0
	    for i in range(drive_resolution_mistiming+1):
	    	solver = SSE_solvers_LCJ_plus_mistiming[i]

	    	if dt + delta_t/2**i < mistiming_phi4:
	    		psi,jumps =  solver.sse_evolve(psi)
	    		norm_sq = block_norm(psi)**2

	    		if len(jumps)>0:    
	    		    jumps[:,0] = jumps[:,0] + t + dt
	    		    jumplist = concatenate((jumplist,jumps))

	    		dt += delta_t/2**i

	    t += dt


	    ###
	    ###    Now do the phi^2 segment
	    ###
	    psi,jumps =  SSE_solver_seg2.sse_evolve(psi)
	    norm_sq = block_norm(psi)**2

	    if len(jumps)>0:    
	        jumps[:,0] = jumps[:,0] + t 
	        jumplist = concatenate((jumplist,jumps))

	            
        # Move forward in time 
	    t+=time_increment_phi2

	    #Do the phi^2 mistiming
	    mistiming_phi2 = 2*delta_t*rng.random() #draw from [0,2*delta_t]
	    dt = 0
	    for i in range(drive_resolution_mistiming+1):
	    	solver = SSE_solvers_LCJ_mistiming[i]

	    	if dt + delta_t/2**i < mistiming_phi2:
	    		psi,jumps =  solver.sse_evolve(psi)
	    		norm_sq = block_norm(psi)**2

	    		if len(jumps)>0:    
	    		    jumps[:,0] = jumps[:,0] + t + dt
	    		    jumplist = concatenate((jumplist,jumps))

	    		dt += delta_t/2**i

	    t += dt


        ###
        ### Now do the normal protocol for a few "cleanup" cycles
        ### 
	    for nperiod in range(N_periods):

	    ## ****  Stabilizer segment  **** ##
	        psi,jumps =  SSE_solver_stab.sse_evolve(psi)

	        if len(jumps)>0:    
	            jumps[:,0] = jumps[:,0] + t 
	            jumplist = concatenate((jumplist,jumps))
	  
            # Move forward in time 
	        t += T_stab


	        #Do the mistiming
	        mistiming_stab = 2*delta_t*rng.random() #draw from [0,2*delta_t]
	        dt = 0
	        for i in range(drive_resolution_mistiming+1):
	        	solver = SSE_solvers_LCJ_mistiming[i]

	        	if dt + delta_t/2**i < mistiming_stab:
	        		psi,jumps =  solver.sse_evolve(psi)
	        		if len(jumps)>0:    
	        		    jumps[:,0] = jumps[:,0] + t + dt
	        		    jumplist = concatenate((jumplist,jumps))

	        		dt += delta_t/2**i

	        t += dt

	        norm_sq = block_norm(psi)**2
	        

        ## ****  Free segment  **** ##

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

	        #Now do the mistiming.
	        #Use the same mistiming (with opposite sign) as in the stabilizer segment
	        mistiming_free = delta_t - mistiming_stab	#the mistiming lies in [-delta_t,delta_t] now
	        dt = 0
	        for i in range(drive_resolution_mistiming):
	        	#Choose the correct SSE solver depending on the sign of mistiming_free
	        	if mistiming_free >= 0:
	        		solver = SSE_solvers_LC_mistiming_pos[i]
	        	else:
	        		solver = SSE_solvers_LC_mistiming_neg[i]

	        	if dt + delta_t/2**(i+1) < abs(mistiming_free):
	        		psi,jumps =  solver.sse_evolve(psi)
	        		if len(jumps)>0:    
	        		    jumps[:,0] = jumps[:,0] + t + dt
	        		    jumplist = concatenate((jumplist,jumps))

	        		dt += delta_t/2**(i+1)

	        t += sign(mistiming_free)*dt 

	        norm_sq = block_norm(psi)**2
	        
        #end loop over driving periods

	    #Calculate quantities of interest for this trajectory 
	    S1s[sample_num]   = LCJ_obj.get_S1_expval(psi)/norm_sq  
	    S2s[sample_num]   = LCJ_obj.get_S2_expval(psi)/norm_sq
	    spins[sample_num] = spin_expectations(psi)
	    S_x,S_y,S_z = spins[sample_num]
	    theta, phi = arccos(S_z), arctan2(S_y,S_x)
	    fidelities[sample_num] = abs(overlap(theta,phi,theta_0,phi_0-phase_sign*pi/8))**2
	    Jump_record.append(jumplist)

    #end loop over SSE trajectories



    # =============================================================================
	# 8. Write the data out to disk

	save_file = data_save_path + "data-omega={}GHz-nu={}-EJ={}GHz-delta_t={}ps-gamma={}GHz-T={}K.dat".format(argv[1],argv[2],argv[3],argv[4],argv[5],argv[6])
	with open(save_file,'wb') as f:
		#Store simulation params and gate time
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