from basic             import 
from units             import *
from numpy             import pi,eye,kron,exp,sqrt,ceil,log,real_if_close
from numpy.linalg      import eigh,norm
from scipy.linalg      import expm
from scipy.special     import jv

from numpy.random      import default_rng 
from gaussian_bath     import bath, get_J_ohmic
from LCJ_circuit       import LCJ_circuit
from SSE_evolver       import sse_evolver

from time import perf_counter
from numba import njit

from struct import pack
from secrets import randbits



def get_delta_operator(Phi_1_A,Phi_1_B,L_1_A,L_2_A,L_1_B,L_2_B,J_c):
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
    n_max = int(ceil(log(eps*(1-r))/log(abs(r))))

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

    # NB: We work in the direct product space of the two qubits. Operators on the combined space
    #   are represented as kron(O_A,O_B) -- i.e., we put the A operators first.

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


    # Construct Hamiltonian and jump operators 
    for n_A in range(0,N_wells_A):

        # Flux operator and JJ potential for qubit A in current A-well
        Phi_1_A = kron(LCJ_obj_A.get_flux_operator_w(n_A-n_0_A),I_B)
        exp_Phi_1_A = expm(2*pi*1j*Phi_1_A/flux_quantum)
        JJ_pot_A = -J_A*(exp_Phi_1_A + exp_Phi_1_A.conj().T)/2

        for n_B in range(0,N_wells_B):

            # Flux operator and JJ potential for qubit B in current B-well
            Phi_1_B = kron(I_A,LCJ_obj_B.get_flux_operator_w(n_B-n_0_B))
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


    # TODO: compute gate time
    T_gate = 1*nanosecond



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



    # TODO: norms!







        


