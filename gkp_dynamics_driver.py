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


"""
Module simulating the dynamics of a driven, dissipative circuit relaizing a GKP qubit.

Author: Liam O'Brien
"""

if __name__ == "__main__":
    from sys import argv
    
    if len(argv) not in [21,22,23]:
        print("Usage: omega E_J/h nu z_s delta_t gamma gamma_q gamma_pL Temp Lambda N_samples N_periods N_wells N_rungs drive_resolution_order output_resolution_order_stab output_resolution_order_free <LCJ_save_path> <data_save_path> <record_wights> seed_SSE (op) seed_init (op)")
        print("Units: \n -> omega in GHz \n -> E_J in GHz \n -> gamma in GHz \n -> gamma_q in e^2/THz \n -> gamma_pL in kHz \n -> Temp in Kelvin \n -> Lambda in gHz \n -> Resolution orders are log2 of number of points desired")
        print("delta_t is (fractional) mistiming of free segment")
        exit(0)

    #Command line arguments
    omega = float(argv[1])*GHz
    E_J = float(argv[2])*GHz*planck_constant
    nu = float(argv[3])
    z_s = float(argv[4])
    delta_t = float(argv[5])
    Gamma = float(argv[6])*GHz
    gamma_q = float(argv[7])*e_charge**2/THz
    gamma_pL = float(argv[8])*kHz
    Temp = float(argv[9])*Kelvin
    Lambda = float(argv[10])*GHz
    N_samples = int(argv[11])
    N_periods = int(argv[12])
    N_wells = int(argv[13])
    N_rungs = int(argv[14])
    drive_resolution_order = int(argv[15]) #Number of times in the first window to sample for binary search
    output_resolution_order_stab = int(argv[16]) #log2(number of points) to output from the stab segment
    output_resolution_order_free = int(argv[17]) #log2(number of points) to output from the free segment
    LCJ_save_path = argv[18]
    data_save_path = argv[19]
    record_weights = int(argv[20])

    if len(argv) >= 22:
        seed_SSE = int(argv[21])
        if len(argv) == 23:
            seed_init = int(argv[22])
        else:
            seed_init = int(randbits(31))
    else:
        seed_SSE = int(randbits(31))
        seed_init = int(randbits(31))



    # =============================================================================
    # 1. Parameters
    Z = nu*Klitzing_constant/4  #LC impedance sqrt{L/C}
    L = Z/omega
    C = 1/(Z*omega) 



    # =============================================================================
    # 2. Initialize quantities from parameters

    #Number of outermost well (=python index of central well)
    n0      = N_wells//2
    parity = -1 if n0 % 2 else 1    #Parity factor to correct sign of spin

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

    # Get LCJ circuit object 
    LCJ_obj = LCJ_circuit(L,C,E_J,(N_wells,N_rungs),mode="0",load_data=1,save_data=1,
        data_path=LCJ_save_path)

    # Hamiltonian (as list of matrices, since its block diagonal)
    H_list = zeros((N_wells,N_rungs,N_rungs),dtype=complex)

    # Energies and eigenvectors, in case we need it
    V_list = zeros((N_wells,N_rungs,N_rungs),dtype=complex)
    E_list = zeros((N_wells,N_rungs),dtype=float)

    # List of jump operators (they are also block-diagonal)
    L_list      = zeros((N_wells,N_rungs,N_rungs),dtype=complex)    #Resistor
    LQ_list      = zeros((N_wells,N_rungs,N_rungs),dtype=complex)   #Charge noise
    L_pL_list = zeros((N_wells,N_rungs,N_rungs),dtype=complex)      #Photon loss

    # Generators for charge noise during free segment
    xi_a_list  = zeros((N_wells,N_rungs,N_rungs),dtype=complex)
    xi_b_list  = zeros((N_wells,N_rungs,N_rungs),dtype=complex)

    # Charge operator (same in every well)
    Q  = 2j*e_charge *LCJ_obj.get_d_dphi_operator_w()
    Xarray = array([Q])         #operator appearing in system-bath coupling (Q for capacitive coupling)


    # Construct Hamiltonian and jump operators in each well
    for n in range(0,N_wells):
        # Get and save Hamiltonian for this well 
        H     = LCJ_obj.H_wl[n]
        H_list[n] = 1*H

        # Get and save flux operator for this well
        Phi = LCJ_obj.get_flux_operator_w(n-n0)

        # Compute and save the resistor jump operators in this well
        [Ld],[E,V]  =  B.get_ule_jump_operator(Xarray, H,return_ed=True)
        L_list[n] = jump_op_coeff*Ld

        #Jump operator for charge noise (proportional to the charge operator Q)
        LQ_list[n] = 1*Q*sqrt(gamma_q)/C
        
        #Jump operator for photon loss (it's proportional to the annihilation operator a)
        #   NB: a = sqrt{\pi/\nu}*(\varphi/\varphi_0 + i\nu q/(2e))
        L_pL_list[n] = sqrt(gamma_pL)*sqrt(pi/nu)*(Phi/flux_quantum + 1.0j*nu/(2*e_charge)*Q)

        # Save energies and eigenvectors in case we need them 
        V_list[n,:,:] = 1*V 
        E_list[n,:]   = 1*E
        
        #Generators for charge noise during free segment
        xi_a = Q + flux_quantum *sqrt(C/L)/(2*pi*e_charge)*Phi
        xi_b = Q - flux_quantum *sqrt(C/L)/(2*pi*e_charge)*Phi
        xi_a_list[n]=xi_a
        xi_b_list[n]=xi_b 


    # Obtain revival time from the smallest spacing of ground state energies (they are nearly commensurate!)
    revival_time = hbar*pi/(2*(-E_list[n0,0]+E_list[n0+1,0]))

    # Define stabilizer segment time, and driving period
    T_s = revival_time*z_s 
    Driving_period = T_s + pi/(2*omega)*(1+delta_t)


    # =============================================================================
    # 5. Initialize SSE solver(s) 

    # Time increment that each instance of sse_evolve evolves over
    time_increment_stab = T_s/(2**output_resolution_order_stab)
    time_increment_free = pi/(2*omega)/(2**output_resolution_order_free)


    # Define inner product, hermitian conjugation, matrix exponentiation and identity operations. 
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

    #Norm for vectors in our grid basis
    block_norm = LCJ_obj.G.norm

    ### Identity operator for block diagonal operators 
    block_id = array([eye(N_rungs,dtype=complex)]*N_wells)

    ### Construct SSE solver for the stabilizer segment
    jump_ops_stab = [L_list,LQ_list]    #NB: no photon loss in the stabilizer segment
    SSE_solver_stab = sse_evolver(H_list,array(jump_ops_stab), time_increment_stab,
        resolution_order=(drive_resolution_order),identity=block_id,hc_method=block_hc,
        dot_method_matrix=block_multiply_matrix,dot_method_vector=block_multiply_vector,
        norm_method = block_norm,expm_method=block_expm,seed=seed_SSE)



    # Set up the SSE solver for the free segment (if necessary)
    # There are four cases:
    #   1) We have charge noise AND photon loss. In this case, we need to resolve the free segment
    #       and construct SSE operators for the full evolution
    #   2) We have ONLY photon loss. In this case, we can perform the dissipative and unitary 
    #       evolution separately. We construct an SSE object for JUST the photon loss, with no 
    #       Hamiltonian (!)
    #   3) We have ONLY charge noise. In this case, we can compute the evolution unitarily, using
    #       analytical results for the charge noise
    #   4) We WANT to resolve the quarter cycle (i.e., output_resolution_free > 0). In this case, 
    #       we need to construct SSE operators for the full evolution 
    # NEW: we also allow for timing noise (via delta_t). If desired, we generate SSE evolution for the 
    #       erroneous segment

    #Diagnose whether we need to resolve the free segement - i.e., are we in case 1 or 4 above?
    resolve_free_segment = (output_resolution_order_free > 0) or ((gamma_pL > 0) and gamma_q > 0)
    timing_noise =  (delta_t != 0)
    
    if resolve_free_segment:
        #Contruct LC Hamiltonian ine each well
        H_LC_list = zeros((LCJ_obj.nwells,LCJ_obj.sample_numgs,LCJ_obj.sample_numgs),dtype=complex)
        for k in range(0,LCJ_obj.nwells):
            H_LC_list[k] = LCJ_obj.compute_H_LC_w(k-LCJ_obj.k0)
        
        # Construct SSE evolution objects for resolving the free segment
        time_increment_free = pi/(2*omega*(2**output_resolution_order_free))
        jump_ops_free = array([LQ_list,L_pL_list])
        SSE_solver_free = sse_evolver(H_LC_list, jump_ops_free,time_increment_free*(1+delta_t),
            resolution_order=drive_resolution_order,identity=block_id,hc_method=block_hc,
            dot_method_matrix=block_multiply_matrix,dot_method_vector=block_multiply_vector,
            norm_method = block_norm,expm_method=block_expm,seed=seed_SSE+1)
    
    elif gamma_pL > 0:
        #Construct SSE object for resolving ONLY the photon loss - with NO system Hamiltonian
        zero_Ham = zeros((LCJ_obj.nwells,LCJ_obj.sample_numgs,LCJ_obj.sample_numgs),dtype=complex)
        jump_ops_free = array([L_pL_list])
        SSE_solver_free = sse_evolver(zero_Ham,jump_ops_free,time_increment_free,
            resolution_order=drive_resolution_order,identity=block_id,hc_method=block_hc,
            dot_method_matrix=block_multiply_matrix,dot_method_vector=block_multiply_vector,
            norm_method = block_norm,expm_method=block_expm,seed=seed_SSE+1)

    #If we're not resolving the free segment and wish to incorporate mistiming, create an additional SSE 
    #   object to resolve the mistiming
    if not resolve_free_segment and timing_noise:
        #Construct LC Hamiltonian ine each well
        H_LC_list = zeros((LCJ_obj.nwells,LCJ_obj.sample_numgs,LCJ_obj.sample_numgs),dtype=complex)
        for k in range(0,LCJ_obj.nwells):
            H_LC_list[k] = LCJ_obj.compute_H_LC_w(k-LCJ_obj.k0)

        # Construct SSE evolution objects for resolving the free segment
        time_increment_free = pi/(2*omega*(2**output_resolution_order_free))
        jump_ops_free = array([LQ_list,L_pL_list])
        SSE_solver_mistiming = sse_evolver(H_LC_list, jump_ops_free,time_increment_free*delta_t,
            resolution_order=drive_resolution_order,identity=block_id,hc_method=block_hc,
            dot_method_matrix=block_multiply_matrix,dot_method_vector=block_multiply_vector,
            norm_method = block_norm,expm_method=block_expm,seed=seed_SSE+1)



    # =============================================================================
    # 6. Set initial state to a random computational state
    
    rng = default_rng(seed_init)

    # Sample bloch angles of qubit
    u , v = rng.uniform(low=0.0,high=1.0,size=2)
    theta , phi = arccos(2*u-1) , 2*pi*v 


    # Construct initial state as
    #   cos(theta/2)|0,0,0>> + exp(i*phi)sin(theta/2)|0,0,1>>
    # We assume TWO logical states encoded in the wells with indices congruent to 0,1 mod nu
    psi0 = zeros((N_wells,N_rungs),dtype=complex)
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
        #   The factor "parity" corrects this
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


    # =============================================================================
    # 7. Solve 

    gauge_optimization_step = 5     #How often to do gauge optimization

    #Lists to store output data in
    S1s = []
    S2s = []
    rung_weights  = []
    well_weights  = []
    # Weights in uppermost rungs (UR) and outermost wells (OW)
    URlist = []
    OWlist = []
    spins  = []     #Spins
    FT_weight_fracs = []    #Fraction of wavefunction norm lost due to truncation from quarter cycle
    Jump_record = []        #Record of quantum jumps occurring
    
    for sample_num in range(0,N_samples):
        # Recorded times 
        tvec   = []
        
        # Spin and stabilizer expectation values
        spins_this_sample  = []
        S2s_this_sample = []
        S1s_this_sample = []
        
        # Weight vs rung index (Rlist) and well index (Wlist)
        Rlist  = []
        Wlist  = []
        
        #Weight of state lost due to truncation after quarter cycles
        FT_weight_list = []
        
        # Lists of quantum jumps 
        jumplist = zeros((0,2))
        
       
        ### Initialize counters 
        
        # Current time
        t = 0
        
        # Current log of norm of state 
        LN = 0 
        
        psi = array(psi0)
        for nperiod in range(N_periods):
            
            ##  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
            # A. Evolve through a stabilizer measurement step
            
            # Iterate over all the steps within the stabilizer measurement that we want to be recorded 
            for step_num in range(0,2**output_resolution_order_stab):
                
                # Evolve state with SSE 
                psi,jumps =  SSE_solver_stab.sse_evolve(psi)
                if step_num % gauge_optimization_step == 0: #Optimize gauge periodically   
                    psinew = LCJ_obj.optimize_gauge(psi)
                    psi = psinew            
                    
                norm_sq = block_norm(psi)**2
                
                if len(jumps)>0:    
                    jumps[:,0] = jumps[:,0] + t 
                    jumplist = concatenate((jumplist,jumps))
                   
                
                # Record stabilizer expectation values 
                S2s_this_sample.append(LCJ_obj.get_S2_expval(psi)/norm_sq)
                S1s_this_sample.append(LCJ_obj.get_S1_expval(psi)/norm_sq)
                
                # Record well and rung weights
                if record_weights:
                    Rlist.append(LCJ_obj.get_rung_weights(psi)/norm_sq)
                    Wlist.append(LCJ_obj.get_well_weights(psi)/norm_sq)

                # Record Expectation values of spin
                spins_this_sample.append(spin_expectations(psi))
                
                # Move forward in time 
                t+=time_increment_stab
           
                
            ##  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
            # B. Evolve psi through quarter cycle at the end of stabilizer measurement 
            #   1) We have charge noise AND photon loss. In this case, we need to resolve the free segment
            #       and construct SSE operators for the full evolution
            #   2) We have ONLY photon loss. In this case, we can perform the dissipative and unitary 
            #       evolution separately. We construct an SSE object for JUST the photon loss, with no 
            #       Hamiltonian (!)
            #   3) We have ONLY charge noise. In this case, we can compute the evolution unitarily, using
            #       analytical results for the charge noise
            #   4) We WANT to resolve the quarter cycle (i.e., output_resolution_free > 0). In this case, 
            #       we need to construct SSE operators for the full evolution 

            # There are three ways this can proceed, depending on whether we're in Case 1 or 4, Case 2, or 
            #   Case 3 (see section 5) 
            if resolve_free_segment:
                #Case 1 or 4: We need to resolve the free segment

                # Iterate over all the steps within the free segment that we want to be recorded 
                for step_num in range(0,2**output_resolution_order_free):  
                    psi,jumps =  SSE_solver_free.sse_evolve(psi)
                    
                    if step_num % gauge_optimization_step == 0:
                        psi = LCJ_obj.optimize_gauge(psi)
                        
                    if len(jumps)>0:    
                        jumps[:,0] = jumps[:,0] + t 
                        jumplist = concatenate((jumplist,jumps))
                    
                    norm_sq = block_norm(psi)**2
                    
                    
                    # Record stabilizer expectation values 
                    S2s_this_sample.append(LCJ_obj.get_S2_expval(psi)/norm_sq)
                    S1s_this_sample.append(LCJ_obj.get_S1_expval(psi)/norm_sq)
                    
                    # Record well and rung weights
                    if record_weights:
                        Rlist.append(LCJ_obj.get_rung_weights(psi)/norm_sq)
                        Wlist.append(LCJ_obj.get_well_weights(psi)/norm_sq)
                    
                    # Record expectations of spin
                    spins_this_sample.append(spin_expectations(psi))
                    
                    # Move forward in time 
                    t += time_increment_free 
         
                    
            else:
                if gamma_pL > 0:
                    #Case 2: We do SSE evolution before applying the quarter cycle unitary
                    psi,jumps =  SSE_solver_free.sse_evolve(psi)
                    if len(jumps)>0:    
                        jumps[:,0] = jumps[:,0] + t 
                        jumplist = concatenate((jumplist,jumps))

                else:
                    #Case 3: We can do everything unitarily

                    #Construct and apply charge unitary (if necessary)
                    if gamma_q > 0:
                        a = rng.normal()*sqrt(pi/4+1/2)/sqrt(omega)
                        b = rng.normal()*sqrt(pi/4-1/2)/sqrt(omega)
                        
                        charge_noise_generator = sqrt(gamma_q/(2*C**2))*(a*xi_a_list + b*xi_b_list) 
                        charge_noise_unitary = block_expm_herm(charge_noise_generator,-1.0j)
                        
                        psi = block_multiply_vector(charge_noise_unitary, psi)
                        
                # Evolve via quarter cycle unitary 
                N_old = block_norm(psi)
                psi = LCJ_obj.optimize_gauge(psi)
                psi = LCJ_obj.apply_quarter_cycle(psi)
                
                # Move forward in time by the time it took to go through a quarter cycle
                t = t + pi/(2*omega)
                
                #Evolve in time if we have mistiming present
                if timing_noise:
                    psi , jumps = SSE_solver_mistiming.sse_evolve(psi)
                    
                    # Record jumps
                    if len(jumps)>0:    
                        jumps[:,0] = jumps[:,0] + t 
                        jumplist = concatenate((jumplist,jumps))
                        t += pi*delta_t/(2*omega)
                
                
                # Record how much weight is lost due to Hilbert space truncation after applying quarter
                #   cycle unitary
                LN += log(block_norm(psi)**2/N_old**2)
                FT_weight_list.append(exp(LN))
                
                # Restore normalization to its pre-quarter cycle value
                psi = psi/block_norm(psi)*N_old
                norm_sq = N_old**2

                
                # Record stabilizer expectation values 
                S2s_this_sample.append(LCJ_obj.get_S2_expval(psi)/norm_sq)
                S1s_this_sample.append(LCJ_obj.get_S1_expval(psi)/norm_sq)
                
                # Record well and rung weights
                if record_weights:
                    Rlist.append(LCJ_obj.get_rung_weights(psi)/norm_sq)
                    Wlist.append(LCJ_obj.get_well_weights(psi)/norm_sq)
                
                # Record Expectation values of spin
                spins_this_sample.append(spin_expectations(psi))

        # end loop over driving periods
            
        ### Record lists 
        S1s.append(array(S1s_this_sample))
        S2s.append(array(S2s_this_sample))
        if record_weights:
            rung_weights.append(array(Rlist))
            well_weights.append(array(Wlist))
        spins.append(array(spins_this_sample))
        FT_weight_fracs.append(array(FT_weight_list))
        Jump_record.append(jumplist)

    #end main loop over SSE trajectories

    # =============================================================================
    # 8. Write the data out to disk

    save_file = data_save_path + "data-omega={}GHz-EJ={}GHz-nu={}-zs={}-delta_t={}-gamma={}GHz-gamma_q={}e^2THz^-1-gamma_pL={}kHz-T={}K.dat".format(argv[1],argv[2],argv[3],argv[4],argv[5],argv[6],argv[7],argv[8],argv[9])
    with open(save_file,'wb') as f:
        #Store simulation params
        params = pack("iiiiiii",N_samples,N_periods,N_wells,N_rungs,output_resolution_order_stab,output_resolution_order_free,record_weights)
        f.write(params)

        num_points = 2**output_resolution_order_stab + 2**output_resolution_order_free
        for i in range(N_samples):
            for j in range(N_periods):
                f.write(pack("f"*num_points,*S1s[i][j*num_points:(j+1)*num_points]))
                f.write(pack("f"*num_points,*S2s[i][j*num_points:(j+1)*num_points]))
                for k in range(num_points):
                    if record_weights:
                        f.write(pack("f"*N_rungs,*rung_weights[i][j*num_points + k]))
                        f.write(pack("f"*N_wells,*well_weights[i][j*num_points + k]))
                    f.write(pack("f"*3,*spins[i][j*num_points + k]))
            
            num_jumps = Jump_record[i].shape[0]
            f.write(pack("i",num_jumps))
            if num_jumps > 0:
                for j in range(num_jumps):
                    f.write(pack("ff",*Jump_record[i][j])) 

            if not resolve_free_segment:
                f.write(pack("f"*N_periods,*FT_weight_fracs[i]))