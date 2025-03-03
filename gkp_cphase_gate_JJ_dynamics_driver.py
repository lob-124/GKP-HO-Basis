from basic             import 
from units             import *
from numpy             import pi,eye,kron,exp,sqrt,ceil,log,real_if_close
from numpy.linalg      import eigh,norm
from numpy.fft         import fft,ifft
from scipy.linalg      import expm
from scipy.special     import jv
from math import isclose

from numpy.random      import default_rng 
from gaussian_bath     import bath, get_J_ohmic
from LCJ_circuit       import LCJ_circuit
from SSE_evolver       import sse_evolver

from time import perf_counter
from numba import njit

from struct import pack
from secrets import randbits



def get_delta_operator(Phi_1_A,Phi_1_B,L_1_A,L_2_A,L_1_B,L_2_B,J_c,tol=1e-15):
    """
    Construct the operator delta = Phi_2^A - Phi_2^B.

    This operator obeys a form of the Kepler equation:
                    
                    M = E - e*sin(E)
        
        where M and e can be expressed in terms of L_1_A, L_2_A, L_1_B, L_2_B.

    We use the series solution for the Kepler equation:

                    E = M + \\sum_{n=1}^{\\infty} 2*J_n(n*e)sin(n*M)/n

        where J_n is the nth Bessel function of the first kind. See 
            https://mathworld.wolfram.com/KeplersEquation.html for details.

    Parameters
    ----------

    Phi_1_A: ndarray(dim_A,dim_A)
            Matrix of the operator Phi_1^A in the current A-well. Should be a kronecker product 
                of Phi_1^A and I_B.
    Phi_1_B: ndarray(dim_B,dim_B)
            Matrix of the operator Phi_1^B in the current B-well. Should be a kronecker product 
                of I_A and Phi_1^B.
    L_1_A: float
            Value of the inductance L_1^A
    L_2_A: float
            Value of the inductance L_2^A
    L_1_B: float
            Value of the inductance L_1^B
    L_1_B: float
            Value of the inductance L_2^B
    J_c: float
            Value of the coupling Josepshon energy J_c
    tol: float, optional
            Desired tolerance for norm of the difference between the returned operator and the "true"
                operator. Use to determine where to cut off the series solution. Defaults to 1e-15


    Returns
    -------
    delta : ndarray(dim_A*dim_B,dim_A*dim_B)
            Operator representing delta = Phi_2^a - Phi_2^B in the current well combination.

    """

    # Constants that M and e can be expressed in terms of
    alpha = 1/L_1_A + 1/L_2_A + 1/L_1_B + 1/L_2_B
    beta  = 1/L_1_A + 1/L_2_A - 1/L_1_B - 1/L_2_B

    # Define M and e for the Kepler equation above. See notes for details
    M = 4*pi/(flux_quantum*L_1_A*(alpha+beta))*Phi_1_A - 4*pi/(flux_quantum*L_1_B*(alpha-beta))*Phi_1_B
    e = - 16*pi**2*alpha*J_c/(flux_quantum**2*(alpha**2-beta**2))

    # The series solution for the Kepler equation converges as a geometric series with ratio r. 
    #   Compute this ratio, and from it determine when to cutoff the sum above.
    if abs(e) >= 1:
        print("Error: solution to Kepler equation will diverge.")
        exit(-1)
    r = e*exp(sqrt(1-e**2))/(1+sqrt(1-e**2))
    n_max = int(ceil(log(tol*(1-r))/log(abs(r))))   # Estimate max index using remainder of geometric series

    # Compute the series solution
    scaled_delta = M
    exp_M = expm(1j*M)
    exp_nM = array(exp_M)
    for n in range(1,n_max+1):
        sin_nM = real_if_close((exp_nM - exp_nM.conj().T)/(2j))
        scaled_delta += 2*jv(n,n*e)*sin_nM/n

        exp_nM = exp_nM @ exp_M

    # The series computes 2*pi*delta/phi_0, so we re-scale to obtain delta
    return scaled_delta*flux_quantum/(2*pi)



def get_logical_state(sigmas_A,sigmas_B,sigmas_AB):
    """
    Construct the logical state of a two-qubit system given the expectations of a subset 
        of the Pauli operators.

    Parameters
    ----------

    sigmas_A: tuple of floats
            Three-tuple (<s_x^A>,<s_y^A>,<s_z^A>) of the logical expectations for qubit A
    sigmas_B: tuple of floats
            Three-tuple (<s_x^B>,<s_y^B>,<s_z^B>) of the logical expectations for qubit B
    sigmas_AB: tuple of floats
            Three-tuple (<s_x^As_x^B>,<s_y^As_y^B>,<s_z^As_z^B>) of the logical expectations 
                on both qubits

    Returns
    -------

    psi: ndarray
            Four-component array with the coefficients of the logical state in the computational basis
    """

    # Unpack the components for convenience
    sx_A , sy_A , sz_A = sigmas_A
    sx_B , sy_B , sz_B = sigmas_B
    sx_AB , sy_AB , sz_AB = sigmas_AB

    # The magnitudes of the four coefficients
    r_00 = sqrt(1+sz_A+sz_B+sz_AB)/2
    r_01 = sqrt(1+sz_A-sz_B-sz_AB)/2
    r_10 = sqrt(1-sz_A+sz_B-sz_AB)/2
    r_11 = sqrt(1-sz_A-sz_B+sz_AB)/2

    # Now compute the sines and cosines of the three phases alpha_01,alpha_10,alpha_11

    c_11 = (sx_AB - sy_AB)/(4*r_11*r_00)

    # We have sin(alpha_11) = +- sqrt(1-cos(alpha_11)^2). We'll try the + sign first one, and if that 
    #   doesn't work we try the - sign.

    try:
        s_11 = sqrt(1-c_11**2)

        c_01 = (r_00*sx_B - r_11*(c_11*sx_A+s_11*sy_A))/(r_10*(sz_A+sz_B))
        s_01 = (r_00*sy_B + r_11*(c_11*sy_A-s_11*sx_A))/(r_10*(sz_A+sz_B))
        c_10 = (r_00*sx_A - r_11*(c_11*sx_B+s_11*sy_B))/(r_10*(sz_A+sz_B))
        s_10 = (r_00*sy_A + r_11*(c_11*sy_B-s_11*sx_B))/(r_10*(sz_A+sz_B))

        eqn = 2*r_11*r_00*c_11 + 2*r_10*r_01*(c_01*c_10 + s_01*s_10)
        assert isclose(eqn,sx_AB)
        
    except:
        s_11 = -sqrt(1-c_11**2)

        c_01 = (r_00*sx_B - r_11*(c_11*sx_A+s_11*sy_A))/(r_10*(sz_A+sz_B))
        s_01 = (r_00*sy_B + r_11*(c_11*sy_A-s_11*sx_A))/(r_10*(sz_A+sz_B))
        c_10 = (r_00*sx_A - r_11*(c_11*sx_B+s_11*sy_B))/(r_10*(sz_A+sz_B))
        s_10 = (r_00*sy_A + r_11*(c_11*sy_B-s_11*sx_B))/(r_10*(sz_A+sz_B))

    
    return array([r_00,r_01*(c_01 + 1j*s_01),r_10*(c_10 + 1j*s_10),r_11*(c_11 + 1j*s_11)])




if __name__ == "__main__":
    from sys import argv
    
    if len(argv) not in [23,24,25]:
        print("Usage: omega_A J_A/h nu_A gamma_A omega_B J_B/h nu_B gamma_B J_C f_A f_B Temp Lambda N_samples N_wells_A N_rungs_A N_wells_B N_rungs_B drive_resolution_order output_resolution_order <LCJ_save_path> <data_save_path> seed_SSE (op) seed_init (op)")
        print("Units: \n -> omega in GHz \n -> J/h in GHz \n -> gamma in GHz \n -> Temp in Kelvin \n -> Lambda in gHz \n -> Resolution orders are log2 of number of points desired")
        print("f_A defines L_1_A and L_2_A via L_1_A = f_A*L_A , L_2_A = (1-f_A)*L_A, and likewise for f_B")
        exit(0)

    #Command line arguments
    omega_A = float(argv[1])*GHz
    J_A = float(argv[2])*GHz*planck_constant
    nu_A = float(argv[3])
    Gamma_A = float(argv[4])*GHz
    omega_B = float(argv[5])*GHz
    J_B = float(argv[6])*GHz*planck_constant
    nu_B = float(argv[7])
    Gamma_B = float(argv[8])*GHz
    J_C = float(argv[9])*GHz*planck_constant
    f_A = float(argv[10])
    f_B = float(argv[11])
    Temp = float(argv[12])*Kelvin
    Lambda = float(argv[13])*GHz
    N_samples = int(argv[14])
    N_wells_A = int(argv[15])
    N_rungs_A = int(argv[16])
    N_wells_B = int(argv[17])
    N_rungs_B = int(argv[18])
    drive_resolution_order = int(argv[19]) #Number of times in the first window to sample for binary search
    output_resolution_order = int(argv[20]) #log2(number of points) to output from the stab segment
    LCJ_save_path = argv[21]
    data_save_path = argv[22]

    if len(argv) >= 24:
        seed_SSE = int(argv[23])
        if len(argv) == 25:
            seed_init = int(argv[24])
        else:
            seed_init = int(randbits(31))
    else:
        seed_SSE = int(randbits(31))
        seed_init = int(randbits(31))



    # =============================================================================
    # 1. Parameters
    Z_A = nu_A*Klitzing_constant/4  #LC impedance sqrt{L/C}
    L_A = Z_A/omega_A
    C_A = 1/(Z_A*omega_A)
    L_1_A , L_2_A = f_A*L_A , (1-f_A)*L_A

    Z_B = nu_B*Klitzing_constant/4  
    L_B = Z_B/omega_B
    C_B = 1/(Z_B*omega_B) 
    L_1_B , L_2_B = f_B*L_B , (1-f_B)*L_B


    # =============================================================================
    # 2. Initialize quantities from parameters

    #Number of outermost well (=python index of central well)
    n0_A      = N_wells_A//2
    parity_A = -1 if n0_A % 2 else 1    #Parity factor to correct sign of spin

    # Vector with well indices
    well_vector_A = arange(-n0_A,n0_A+1)

    # Repeat for qubit B
    n0_B      = N_wells_B//2
    parity_B = -1 if n0_B % 2 else 1 
    well_vector_B = arange(-n0_B,n0_B+1)


    # =============================================================================
    # 3. Initialize bath
    #		We assume the bath is the same for both qubits - but the coupling may not be!

    # Generate spectral function for an ohmic bath
    spectral_function = get_J_ohmic(Temp, Lambda,omega0=1)

    # Generate bath object
    B = bath(spectral_function,5*Lambda,Lambda/1e5)

    #Compute 1/(hbar*C_R) from Gamma and J(\epsilon_0/\hbar) for each qubit
    epsilon_0_A = sqrt(4*e_charge**2*J_A/C_A)
    J_0_A = spectral_function(epsilon_0_A/hbar)
    _lambda_A = (nu_A*planck_constant*omega_A/J_A)**0.25/(2*pi)
    jump_op_coeff_A = sqrt(pi*Gamma_A/J_0_A)*_lambda_A/e_charge

    epsilon_0_B = sqrt(4*e_charge**2*J_B/C_B)
    J_0_B = spectral_function(epsilon_0_B/hbar)
    _lambda_B = (nu_B*planck_constant*omega_B/J_B)**0.25/(2*pi)
    jump_op_coeff_B = sqrt(pi*Gamma_B/J_0_B)*_lambda_B/e_charge




    # =============================================================================
    # 4. Initialize system

    # NB: We work in the direct product space of the two qubits. 
    # Block-diagonal operators on the combined space are represented as 
    #   kron(O_A,O_B) -- i.e., we put the A operators first.
    # In general, operators have four indices: O(N_A,N_B,nu',nu) is the matrix element in 
    #   A-well N_A and B-well N_B that goes from nu -> nu', where nu',nu are "combined" indices
    #   of the Kronencker product

    # Get LCJ circuit objects for each qubit 
    LCJ_obj_A = LCJ_circuit(L_A,C_A,J_A,(N_wells_A,N_rungs_A),mode="0",load_data=1,save_data=1,
        data_path=LCJ_save_path)
    LCJ_obj_B = LCJ_circuit(L_B,C_B,J_B,(N_wells_B,N_rungs_B),mode="0",load_data=1,save_data=1,
        data_path=LCJ_save_path)

    # Hamiltonian (as list of matrices, since its block diagonal)
    H_list_AB = zeros((N_wells_A,N_wells_B,N_rungs_A*N_rungs_B,N_rungs_A*N_rungs_B),dtype=complex)

    # # Energies and eigenvectors, in case we need it
    # E_list_A = zeros((N_wells_A,N_rungs_A),dtype=float)
    # E_list_B = zeros((N_wells_B,N_rungs_B),dtype=float)

    # List of jump operators. There are two: one arising from coupling to each qubit 
    L_list_A      = zeros((N_wells_A,N_wells_B,N_rungs_A*N_rungs_B,N_rungs_A*N_rungs_B),dtype=complex)
    L_list_B      = zeros((N_wells_A,N_wells_B,N_rungs_A*N_rungs_B,N_rungs_A*N_rungs_B),dtype=complex)

    # Identity operators on the two qubits. Used to upgrade single-qubit operators to operators on the 
    #   combined Hilbert space
    I_A = eye(*Phi_1_A.shape)
    I_B = eye(*Phi_1_B.shape)

    # Charge operators for each qubit (same in all wells)
    Q_A  = 2j*e_charge *LCJ_obj_A.get_d_dphi_operator_w()
    cap_term_A = Q_A @ Q_A /(2*C_A)
    Q_B  = 2j*e_charge *LCJ_obj_B.get_d_dphi_operator_w()
    cap_term_B = Q_B @ Q_B  / (2*C_B)
    Xarray = array([kron(Q_A,I_B) , kron(I_A,Q_B)]) # Coupling terms between each qubit and the bath


    # Logical z operators for the two qubits
    sigma_z_A = zeros((N_wells_A,N_wells_B,N_rungs_A*N_rungs_B,N_rungs_A*N_rungs_B),dtype=complex)
    sigma_z_B = zeros((N_wells_A,N_wells_B,N_rungs_A*N_rungs_B,N_rungs_A*N_rungs_B),dtype=complex)
    sigma_Z_AB = zeros((N_wells_A,N_wells_B,N_rungs_A*N_rungs_B,N_rungs_A*N_rungs_B),dtype=complex)


    # Construct Hamiltonian and jump operators 
    for n_A in range(0,N_wells_A):

        # Flux operator and JJ potential for qubit A in current A-well
        Phi_1_A = kron(LCJ_obj_A.get_flux_operator_w(n_A-n0_A),I_B)
        exp_Phi_1_A = expm(2*pi*1j*Phi_1_A/flux_quantum)
        JJ_pot_A = -J_A*(exp_Phi_1_A + exp_Phi_1_A.conj().T)/2

        for n_B in range(0,N_wells_B):

            # Flux operator and JJ potential for qubit B in current B-well
            Phi_1_B = kron(I_A,LCJ_obj_B.get_flux_operator_w(n_B-n0_B))
            exp_Phi_1_B = expm(2*pi*1j*Phi_1_B/flux_quantum)
            JJ_pot_B = -J_B*(exp_Phi_1_B + exp_Phi_1_B.conj().T)/2


            # Construct the operators delta = Phi_2^A - Phi_2^B and Sigma = Phi_2^A + Phi_2^B
            delta = get_delta_operator(Phi_1_A,Phi_1_B,L_1_A,L_2_A,L_1_B,L_2_B,J_c)
            Sigma = (2/(alpha*L_1_A))*Phi_1_A + (2/(alpha*L_1_B))*Phi_1_B

            # Construct Phi_2^A and Phi_2^B out of Sigma and delta
            Phi_2_A = (Sigma + delta)/2
            Phi_2_B = (Sigma - delta)/2

            # Construct the coupling JJ potential
            exp_coupler = expm(2*pi*1j*delta/flux_quantum)
            JJ_pot_coupler = -J_c*(exp_coupler + exp_coupler.conj().T)/2

            # Construct the Hamiltonian
            _H = (Phi_1_A-Phi_2_A)@(Phi_1_A-Phi_2_A)/(2*L_1_A) + Phi_2_A@Phi_2_A/(2*L_2_A) + JJ_pot_A
                    + (Phi_1_B-Phi_2_B)@(Phi_1_B-Phi_2_B)/(2*L_1_B) + Phi_2_B@Phi_2_B/(2*L_2_B) + JJ_pot_B
                    + JJ_pot_coupler + cap_term_A + cap_term_B
            H_list_AB[n_A,N_B] = array(_H)

            # Construct the jump operators
            [L_A,L_B] = B.get_ule_jump_operator(Xarray,_H)
            L_list_A[n_A,n_B] = jump_op_coeff_A*array(L_A)
            L_list_B[n_A,n_B] = jump_op_coeff_B*array(L_B)


            # Construct the sigma_z spin operators
            sigma_z_A[n_A,n_B] = kron(LCJ_obj_A.sz_wl[n_A],I_B)
            sigma_z_B[n_A,n_B] = kron(I_A,LCJ_obj_B.sz_wl[n_B])
            sigma_Z_AB[n_A,n_B] = kron(LCJ_obj_A.sz_wl[n_A],LCJ_obj_B.sz_wl[n_B])


    # TODO: compute gate time
    T_gate = 1*nanosecond
    revival_time_A = 1/omega_A
    revival_time_B = 1/omega_B



    # =============================================================================
    # 5. Initialize SSE solver(s) 

    # Time increment that each instance of sse_evolve evolves over
    time_increment_gate = T_gate/(2**output_resolution_order_stab)


    # Define inner product, hermitian conjugation, matrix exponentiation and identity operations on the 
    #   product Hilbert space, exploiting the block-diagonal structure

    @njit
    def block_multiply_matrix(A,B):
        """
        Matrix product for block-diagonal matrices A,B on the product Hilbert space
        """
        out = zeros((A.shape[0],A.shape[1],A.shape[2],B.shape[3]),dtype=complex128)
        for i in range(A.shape[0]):
            for j in range(A.shape[1])
                out[i,j] = A[i,j] @ B[i,j]
        return out
       

    @njit
    def block_multiply_vector(A,B):
        """
        Matrix product for block-diagonal matrix A and block vector B  on the product Hilbert space
        """
        out = zeros((A.shape[0],A.shape[1],A.shape[2]),dtype=complex128)
        for i in range(A.shape[0]):
            for j in range(A.shape[1])
                out[i,j] = A[i,j] @ B[i,j]
        return out
        
    @njit
    def block_hc(mat):
        """
        Hermitian conjugate for block-diagonal matrix on the product Hilbert space
        """
        return conj(mat.transpose((0,1,3,2)))

    def block_expm(mat):
        """
        Matrix exponential of block-diagonal matrix on the product Hilbert space
        """
        out = zeros(mat.shape,dtype=complex128)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                out[i,j] = expm(mat[i,j])
            
        return out 

    @njit
    def block_expm_herm(mat,alpha):
        """
        Matrix exponential of block-diagonal (hermitian) matrix on the product Hilbert space

        NB: diagonalization is faster than expm() for hermitian matrices with jit'ing
        """
        out = zeros(mat.shape,dtype=complex128)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                evals,evecs = eigh(mat[i,j])
                out[i,j] = evecs @ diag(exp(alpha*evals).astype(complex128)) @ evecs.conj().T
            
        return out 

    @njit
    def apply_k_array(O,psi):
        """
        Apply the k-space operator O (acting on both qubits) to the vector v.
        Note that psi is assumed NOT to be in k-space.
        """
        fftpsi = fft(fft(psi,axis=0),axis=1)
        Y = block_multiply_vector(O,fftpsi)
        return ifft(ifft(Y,axis=1),axis=0)


    # Define inner product and norms on the combined Hilbert space

    # Construct composite overlap matrices
    overlap_A_shape = LCJ_obj_A.G.overlap_matrices_k.shape
    overlap_B_shape = LCJ_obj_B.G.overlap_matrices_k.shape
    overlap_combined = zeros((overlap_shape_A[0],overlap_shape_B[0],overlap_shape_A[1]*overlap_shape_B[1],
                            overlap_shape_A[2]*overlap_shape_B[2]),dtype=complex128)
    for i in range(overlap_shape_A[0]):
        for j in range(overlap_shape_B[0]):
            overlap_combined[i,j] = kron(LCJ_obj_A.G.overlap_matrices_k[i],LCJ_obj_B.overlap_matrices_k[j])

    @njit
    def block_inner_product(psi2,psi1):
        Vpsi1 = apply_k_array(overlap_combined,psi1)
        return sum(psi2.conj()*Vpsi1)

    @njit
    def block_norm(psi):
        return real(sqrt(block_inner_product(v,v)))


    # Block expectation of an operator
    @njit 
    def block_expectation(O,psi):
        return block_inner_product(psi,block_multiply_vector(O,psi))


    ### Construct SSE solver for the gate segment
    jump_ops_stab = [L_list_A,L_list_B]
    SSE_solver_stab = sse_evolver(H_list,array(jump_ops_stab), time_increment_stab,
        resolution_order=drive_resolution_order,hc_method=block_hc,
        dot_method_matrix=block_multiply_matrix,dot_method_vector=block_multiply_vector,
        norm_method = block_norm,expm_method=block_expm,seed=seed_SSE)



    # =============================================================================
    # 6. Set the initial state of each qubit
    
    rng = default_rng(seed_init)

    # Sample bloch angles of each qubit
    u_A , v_A = rng.uniform(low=0.0,high=1.0,size=2)
    theta_A , phi_A = arccos(2*u_A-1) , 2*pi*v_A
    u_B , v_B = rng.uniform(low=0.0,high=1.0,size=2)
    theta_B , phi_B = arccos(2*u_B-1) , 2*pi*v_B 


    # Construct initial state as
    #   cos(theta/2)|0,0,0>> + exp(i*phi)sin(theta/2)|0,0,1>>
    # We assume TWO logical states encoded in the wells with indices congruent to 0,1 mod nu
    psi0 = zeros((N_wells_A,N_wells_B,N_rungs_A*N_rungs_B),dtype=complex)
    for n_A in range(N_wells_A):
        well_index_A = n_A - n0_A
        if well_index_A % nu_A == 0:
            psi_A = cos(theta_A/2)*exp(-(well_index_A)**2*LCJ_obj_A.sigma**2/(8*LCJ_obj_A.r**2))
        elif well_index_A % nu_A == 1:
            psi_A = sin(theta/2)*exp(1j*phi_A)*exp(-(well_index_A)**2*LCJ_obj_A.sigma**2/(8*LCJ_obj_A.r**2))

        for n_B in range(N_wells_B):
            well_index_B = n_B - n0_B        
            if well_index_B % nu_B == 0:
                psi0[n_A,n_B,0] = psi_A*cos(theta_B/2)*exp(-(well_index_B)**2*LCJ_obj_B.sigma**2/(8*LCJ_obj_B.r**2))
            elif well_ind % nu == 1:
                psi0[n_A,n_B,0] = psi_A*sin(theta_B/2)*exp(1j*phi_B)*exp(-(well_index_B)**2*LCJ_obj_B.sigma**2/(8*LCJ_obj_B.r**2))
     
    psi0 = psi0/block_norm(psi0)



    # Define functions for the spin expectations 

    # Functions for the quarter cycle evolution on each qubit separately. Assumed to be the identity on the
    #   other qubit
    QC_mat_A = LCJ_obj_A.quarter_cycle_matrix.reshape(N_wells_A,N_rungs_A,N_wells_A,N_rungs_A)
    def apply_QC_A(psi):
        vec = psi.reshape(N_wells_A,N_wells_B,N_rungs_A,N_rungs_B)
        return einsum("abcd,cedf->aebf",QC_mat_A,vec).reshape(N_wells_A,N_wells_B,N_rungs_A*N_rungs_B)

    QC_mat_B = LCJ_obj_B.quarter_cycle_matrix.reshape(N_wells_B,N_rungs_B,N_wells_B,N_rungs_B)
    def apply_QC_B(psi):
        vec = psi.reshape(N_wells_A,N_wells_B,N_rungs_A,N_rungs_B)
        return einsum("abcd,ecfd->eafb",QC_mat_B,vec).reshape(N_wells_A,N_wells_B,N_rungs_A*N_rungs_B)


    # Generate S-gate matrices (used in computing sigma_y)
    S_mats_A = zeros((N_wells_A,N_wells_B,N_rungs_A*N_rungs_B,N_rungs_A*N_rungs_B),dtype=complex128)
    for n_A in range(0,N_wells_A):
        well_ind = n - n0_A
        H = LCJ_obj_A.get_H_w(well_ind)
        S_mats_A[n_A] = kron(expm(-1j*H*revival_time_A/hbar),I_A)

    S_mats_B = zeros((N_wells_B,N_wells_B,N_rungs_B*N_rungs_B,N_rungs_B*N_rungs_B),dtype=complex128)
    for n_B in range(0,N_wells_B):
        well_ind = n - n0_B
        H = LCJ_obj_B.get_H_w(well_ind)
        S_mats_B[n_B] = kron(I_A,expm(-1j*H*revival_time_B/hbar))


    def spin_expectations_A(psi):
        #NB! The spin has the wrong sign if n0 % 2 == 1.
        #   The factor "parity" corrects this
        norm_sq = block_norm(psi)**2

        S_z = parity_A*block_expectation(sigma_z_A,psi)/norm_sq
        
        #Compute <S_x> by applying Hadamard then computing <S_z>
        psi_x = apply_QC_A(psi)
        S_x = parity_A*block_expectation(sigma_z_A,psi_x)/norm_sq
        
        #Compute <S_y> by applying S gate, then Hadamard, then computing <S_z>
        psi_y = zeros((N_wells_A,N_wells_B,N_rungs_A*N_rungs_B),dtype=complex)
        for n_A in range(N_wells_A):
            for n_B in range(N_wells_B)
                psi_y[n_A,n_B] = S_mats_A[n_A] @ psi[n_A,n_B]
        psi_y =  apply_QC_A(psi_y)
        S_y = parity_A*block_expectation(sigma_z_A,psi_y)/norm_sq
        
        return S_x , S_y, S_z

    def spin_expectations_B(psi):
        #NB! The spin has the wrong sign if n0 % 2 == 1.
        #   The factor "parity" corrects this
        norm_sq = block_norm(psi)**2

        S_z = parity_B*block_expectation(sigma_z_B,psi)/norm_sq
        
        #Compute <S_x> by applying Hadamard then computing <S_z>
        psi_x = apply_QC_B(psi)
        S_x = parity_B*block_expectation(sigma_z_B,psi_x)/norm_sq
        
        #Compute <S_y> by applying S gate, then Hadamard, then computing <S_z>
        psi_y = zeros((N_wells_A,N_wells_B,N_rungs_A*N_rungs_B),dtype=complex)
        for n_A in range(N_wells_A):
            for n_B in range(N_wells_B)
                psi_y[n_A,n_B] = S_mats_B[n_B] @ psi[n_A,n_B]
        psi_y =  apply_QC_B(psi_y)
        S_y = parity_B*block_expectation(sigma_z_B,psi_y)/norm_sq
        
        return S_x , S_y, S_z

    def spin_expectations_AB(psi):
        #NB! The spin has the wrong sign if n0 % 2 == 1.
        #   The factor "parity" corrects this
        norm_sq = block_norm(psi)**2

        S_z = parity_A*parity_B*block_expectation(sigma_z_AB,psi)/norm_sq
        
        #Compute <S_x> by applying Hadamard then computing <S_z>
        psi_x = apply_QC_B(psi)
        psi_x = apply_QC_A(psi)
        S_x = parity_A*parity_B*block_expectation(sigma_z_AB,psi_x)/norm_sq
        
        #Compute <S_y> by applying S gate, then Hadamard, then computing <S_z>
        psi_y = zeros((N_wells_A,N_wells_B,N_rungs_A*N_rungs_B),dtype=complex)
        for n_A in range(N_wells_A):
            for n_B in range(N_wells_B)
                psi_y[n_A,n_B] = S_mats_A[n_A] @ S_mats_B[n_B] @ psi[n_A,n_B]
        psi_y =  apply_QC_B(psi_y)
        psi_y = apply_QC_A(psi_y)
        S_y = parity_A*parity_B*block_expectation(sigma_z_AB,psi_y)/norm_sq
        
        return S_x , S_y, S_z









        


