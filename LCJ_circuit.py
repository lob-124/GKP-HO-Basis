#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convention for matrix represenations:
    
_wl = block diagonal matrix in well space                     (nwell,nrungs,nrungs)
_w  = acting on single well                                   (nrungs,nrungs)
_k  = fourier space                                           (nwell,nrungs,nrungs)
_b  = off-diagonal blocks                                     (2*n+1,nrungs,nrungs) (for translationally-invariant matrices)
_f  = full matrix
phi = self.sigma * x 

"""
from basic import *
from numpy import *
from numpy.linalg import *
from scipy.special import factorial,hermite,jv
from units import *
import scipy.sparse as sp
import scipy.sparse.linalg as spla 
from quarter_cycle_in_grid_basis import get_quarter_cycle_matrix
from HO_grid import ho_grid
from numpy.fft import fft,ifft

from numba import njit



class LCJ_circuit():
    """
    A class representing an LCJ circuit
    """
    def __init__(self,L,C,Josephson_energy,grid_shape,mode="0",save_data=True,load_data=True,data_path=None,step_function_order=200):
        self.L = L
        self.C = C
        self.J = Josephson_energy
        
        self.step_function_order = 100
        self.shape  = grid_shape
        self.nwells = grid_shape[0]
        self.nrungs = grid_shape[1]

        if mode == "pi":
            self.mode = 1 
            assert self.nwells%2 ==0,"nwells must be an even integer in pi mode"
        elif mode == "0":
            self.mode = 0
            assert self.nwells%2 ==1,"nwells must be an odd integer in 0 mode"
        else:
            raise ValueError(f"Mode {mode} not recognized. Mode must be either '0' or 'pi'")
        

        # The LC impedance and frequency
        self.Z= sqrt(L/C)
        self.nu = int(round(4*e_charge**2*self.Z/planck_constant))
        self.omega = 1/sqrt(L*C)
        
        # Parameters of the grid
        self.r = 0.5+sqrt(0.25-self.omega/(4*pi*self.J))
        self.well_spacing = 2*pi*self.r 
        self.sigma = sqrt(2)*self.r*(pi*hbar*self.omega/self.J)**(1/4)
        self.S2_revival_time = self.r/self.omega 
       
        self.D = self.nwells*self.nrungs    
       
        # Well index of the rightmost well in the simulation (=python index of central well)
        self.k0 = (self.nwells- (not self.mode))//2

        self.params = array([self.nwells,self.nrungs,self.L,self.C,self.J,self.mode])
        self.paramstr=f"LCJ_data__{self.nwells}_{self.nrungs}_{int(self.L/(1e-9*Henry))}"+
                       f"_{int(self.C/(1e-15*Farad))}_{int(self.J/GHz*1e3)}_{self.mode}.npz"
        if data_path is None:
            self.DataPath = "../Data/LCJ_circuits/"+self.paramstr
        else:
            self.DataPath = data_path + self.paramstr
        
        self.loaded_data = False
        if load_data:
            #Attempt to load an LCJ_Circuit object for these parameters from disk
            try:
                DataPath   = self.DataPath
                Data       = load(DataPath,allow_pickle=True)
                LCJ        = Data["LCJ_circuit"].item()
                data_params = Data["params"]
                
                # HO grid object and block matrices
                self.G = LCJ.G
                self.quarter_cycle_matrix = LCJ.quarter_cycle_matrix
                S = eigvals(self.quarter_cycle_matrix)
                self.H_wl = LCJ.H_wl
                self.S1_wl = LCJ.S1_wl
                self.sz_wl = LCJ.sz_wl
                self.quarter_cycle_matrix = LCJ.quarter_cycle_matrix
                self.sqsc = LCJ.sqsc 

                assert prod([abs(data_params[n]-self.params[n])<1e-10 for n in range(0,len(self.params))])

                print(f"Sucessfully loaded data from file {DataPath}")
                
                self.loaded_data = True 
            
            except FileNotFoundError:
                pass 

        if self.loaded_data ==False:
            #If loading from disk fails, create the desired LCJ_Circuit object
            
            print(f"No LCJ circuit found with the requested parameters. Generating from new")
            
            # HO grid object and block matrices
            grid_spacing  =  self.well_spacing/self.sigma
            self.G = ho_grid(self.nwells, self.nrungs, grid_spacing,offset=0.5*grid_spacing*self.mode)
            self.H_wl = self.compute_H_wl()
            self.S1_wl = self.compute_stabilizer_1_wl()
            self.sz_wl = self.compute_sz_wl()
            self.quarter_cycle_matrix = self.compute_quarter_cycle_matrix()
            self.loaded_data = False 
            self.sqsc = self.compute_sqsc()
            
                                
        self.overlap_matrices = self.G.overlap_matrices
        
        assert len(self.overlap_matrices)%2 == 1 
        
        self.n0 = (shape(self.overlap_matrices)[0]-1)//2 
        if self.n0 > 0:        
            self.overlap_norm = [norm(self.overlap_matrices[self.n0+1][:k,:k]) for k in range(0,self.nrungs)]
        else:
            self.overlap_norm = zeros((self.nrungs))

        if save_data and not self.loaded_data:
            savez(self.DataPath,params = self.params,LCJ_circuit = self,allow_pickle=True)


    def compute_quarter_cycle_matrix(self):
        """
        Computes the quarter cycle matrix (QCM) - i.e., the unitary representing evolution by 
            1/4 of an LC period under the LC Hamiltonian
        """
        # Compute the matrix containing the action of the QCM on the grid states
        F0     = get_quarter_cycle_matrix((self.nwells,self.nrungs), self.sigma, self.r)
        F0_r   = F0.reshape((self.nwells,self.nrungs,self.nwells,self.nrungs))


        # To get the QCM in the grid basis, we have to account for the fact that the grid basis
        #   is non-orthogonal. This essentially entails multiplying by the inverse of the overlap 
        #   matrix.
        # The overlap matrix is translationally invariant in the well index, so it's computationally
        #   easier to Fourier Transform with respect to the well index. In Fourier space, the overlap 
        #   matrix is diagonal and easily diagonalized.

        # Fourier Transform and apply inverse overlap matrix
        F0_r_fft = fft(F0_r,axis=0)
        QCM_r_fft = einsum("abc,acde->abde",self.G.V_inv_k,F0_r_fft)
        
        # Optimize gauge 
        QCM_r_fft = einsum("abc,acde->abde",self.G.gauge_optimizer_k,QCM_r_fft)
        
        #Inverse Fourier Transform and re-shape to be block-diagonal
        QCM_r = ifft(QCM_r_fft,axis=0)
        QCM = QCM_r.reshape((self.nwells*self.nrungs,self.nwells*self.nrungs))

        return  QCM
          

    def optimize_gauge(self,psi):
        """
        Function that optimizes the gauge of the given wavefunction.

        The grid basis is overcomplete, so there is a "gauge freedom" in how to represent 
            a given physical state. We choose the "gauge" that minimizes the expected value 
            of the "rung index".
        """
        return self.G.optimize_gauge(psi)
      
        
    def apply_quarter_cycle(self,psi):
        """
        Apply the quarter cycle matrices to a given state
        """
        out = (self.quarter_cycle_matrix @ psi.flatten()).reshape((self.nwells,self.nrungs))
        return out
    

    def get_phi_operator_w(self,k,dim="default",format="array"):
        """
        Compute the phase operator Phi in well k

        Parameters
        ----------
        k : int
            index of well, for which we use eigenstates.
        dim : int, optional
            Dimension of the operator. The default is self.nrungs.

        Returns
        -------
        Phi : sp.csc(dim,dim,dtype=complex)
            Operator representing Phi in the eigenbasis of well <well_number>.

        """
        Phi = self.G.get_x_operator_in_well_basis(k,dim=dim,format=format)*self.sigma 
        return Phi


    def get_phi2_operator_w(self,k,L_1,L_2,E_J_prime,eps=1e-10,dim="default",format="array"):
        """
        Compute the phase operator phi_2, represented in the basis of well k

        Phi_2 is defined in terms of phi_1 via the equation
                phi_2 + 4*pi^2*E_J'*L_1*L_2/(Phi_0^2*(L_1+L_2))*sin(phi_2) = (L_2/(L_1_L_2))*phi_1
        This is an example of Kepler's equation, which has a series solution we make use of.


        Parameters
        ----------
        k : int
            index of well, for which we use eigenstates.
        dim : int, optional
            Dimension of the operator. The default is self.nrungs.

        Returns
        -------
        phi_2 : sp.csc(dim,dim,dtype=complex)
            Operator representing phi_2 in the eigenbasis of well <well_number>.

        """ 
        if dim == "default":
            dim = self.nrungs

        #We seek to solve an example of Kepler's equation E - e*Sin(E) = M
        # This equation has series solution:
        #           E = M + \sum_{n=1}^{\infty} (2/n)*J_n(ne)*sin(nM)
        #    with J_n the nth Bessel fucntion of the first kind.
        #
        # See https://mathworld.wolfram.com/KeplersEquation.html for details.

        #Define e here
        e = -4*pi**2*E_J_prime*L_1*L_2/(flux_quantum**2*(L_1+L_2))

        #The above series converges as a geometric series with ratio
        #           r = e/(1+sqrt(1-e^2))*exp(sqrt(1-e^2))
        # We use this r, and the known formula for the remainder of a geometric series, to estimate
        #    the n at which to cut off the above sum for the desired accruacy eps. 
        r = e/(1+sqrt(1-e**2))*exp(sqrt(1-e**2))
        n_max = int(ceil(log(eps*(1-r))/log(r)))

        #Define M here
        phi_1 = self.G.get_x_operator_in_well_basis(k,dim=dim+n_max,format=format)*self.sigma #Dimensionless phase phi_1 = Phi_1*2*pi/Phi_0
        M = L_2/(L_1+L_2)*phi_1

        phi_2 = array(M)    #NB: CHECK DATA TYPE. DO WE NEED COMPLEX HERE?
        exp_i_M = expm(1j*M)
        for n in range(1,n_max+1):
            #Compute the operator sin(nM)
            if n == 1:
                _exp = exp_i_M
            else:
                _exp = _exp @ exp_i_M
            sin_nM = (_exp + _exp.conj().T)/(2j)

            phi_2 += (2/n)*jv(n,n*e)*sin_nM
        
        return phi_2 


    def get_d_dphi_operator_w(self,dim="default",format="array"):
        """
        Compute the operator \\partial_phi
        NB: \\partial_phi is independent of the well index
    
        Parameters
        ----------
    
        dim : int, optional
            Dimension of the operator. The default is self.nrungs.
    
        Returns
        -------
        dPhi : sp.csc(dim,dim,dtype=complex)
            Operator representing \partial_phi in the eigenbasis of the well 
    
        """
        ddPhi = self.G.get_ddx_operator_in_well_basis(dim=dim,format=format)/self.sigma 
        return ddPhi 
  

    def get_q_operator_w(self,dim="default",format="array"):
        """
        Compute the charge operator q 
        NB: q is independent of the well index
    
        Parameters
        ----------
        
        dim : int, optional
            Dimension of the operator. The default is self.nrungs.
    
        Returns
        -------
        Phi : sp.csc(dim,dim,dtype=complex)
            Operator representing Phi in the eigenbasis of well <well_number>.
    
        """
        ddphi = self.get_d_dphi_operator_w(dim=dim,format=format)
        return -2*1j*e_charge*ddphi

    
    def get_flux_operator_w(self,k,dim="default",format="array"):
        """
        Compute the flux operator in well k

        Parameters
        ----------
        k : int
            index of well, for which we use eigenstates.
        dim : int, optional
            Dimension of the operator. The default is self.nrungs.

        Returns
        -------
        Flux : sp.csc(dim,dim,dtype=complex)
            Operator representing Flux in the eigenbasis of well <well_number>.

        """
        Phi = self.get_phi_operator_w(k,dim=dim,format=format)
        Flux = Phi*flux_quantum/(2*pi)
        return Flux 


    def compute_H_LC_w(self,k,dim = "nrungs"):
        """
        Compute matrix representation of LC Hamiltonian in well k

        Parameters
        ----------
        k : int
            Well index requested
        dim : int, optional
            Number of rungs included. The default is self.nrungs.

        Returns
        -------
        H_LC : ndarray((dim,dim),complex)
            Matrix representation of H_LC in well k.

        """

        if dim=="nrungs":
            dim=self.nrungs 
        
        Q    = self.get_q_operator_w(dim=dim+2,format="array")
        Flux  = self.get_flux_operator_w(k,dim=dim+2,format="array")
        
        HC  =  Q@Q / (2*self.C) 
        HL  =  Flux@Flux / (2*self.L)
        H_LC = HC+HL
    
        return H_LC[:dim,:dim]    
        

    def compute_H_JJ_w(self,k,dim="nrungs"):
        """
        Compute matrix representation of JJ Hamiltonian in well k

        Parameters
        ----------
        k : int
            Well index requested
        dim : int, optional
            Number of rungs included. The default is self.nrungs.

        Returns
        -------
        H_JJ : ndarray((dim,dim),complex)
            Matrix representation of H_JJ in well k.
        """
        if dim=="nrungs":
            dim = self.nrungs
        
        # Compute the dimension to keep the matrices to
        max_order = get_maximal_order_of_exponential(dim, 1,sigma=self.sigma)
        D = dim + max_order 
        
        # Compute cos(phi)
        exp_i_phi = self.G.get_exp_alpha_x_operator(1j*self.sigma, k)
        cos_phi  = 0.5*(exp_i_phi+exp_i_phi.conj().T)
        
        return self.J*cos_phi*(-1**(1-self.mode))
        

    def compute_stabilizer_1_wl(self):
        """
        Compute stabilizer 1, as block-diagonal matrix in well basis
        
        Returns
        -------
        S1_wl : ndarray((nwells,nrungs,nrungs),complex)
            Stabilizer 1, as a Block diagonal matrix in well space.

        """
        alpha = 2*pi*self.sigma/self.well_spacing
        S1_wl = array([self.G.get_exp_alpha_x_operator(2*pi*1j*self.sigma/self.well_spacing, k) for k in range(-self.k0,self.k0+1)])
        return S1_wl
        

    def get_exp_alpha_phi_operator(self,alpha,k,dim="nrungs"):
        """
        Compute matrix representing e^{alpha*phi} in well k
        """
        return self.G.get_exp_alpha_x_operator(alpha*self.sigma, k,dim=dim)
    

    def get_exp_alpha_ddphi_operator(self,alpha):
        """
        Compute matrix representing e^{alpha*ddphi} in well k
        """
        return self.G.get_exp_alpha_ddx_operator(alpha/self.sigma)
    

    def compute_H_JJ_w(self,k,dim="nrungs"):
        """
        Compute matrix representation of JJ Hamiltonian in well k

        Parameters
        ----------
        k : int
            Well index requested
        dim : int, optional
            Number of rungs included. The default is self.nrungs.

        Returns
        -------
        H_JJ : ndarray((dim,dim),complex)
            Matrix representation of H_JJ in well k.
        """
        if dim=="nrungs":
            dim = self.nrungs
            
        max_order = get_maximal_order_of_exponential(dim, 1,sigma=self.sigma)
        D = dim + max_order 
        
        exp_i_phi = self.G.get_exp_alpha_x_operator(1j*self.sigma, k)
        cos_phi  = 0.5*(exp_i_phi+exp_i_phi.conj().T)
        
        return self.J*cos_phi*(-1**(1-self.mode))
        

    def compute_sqsc_in_well(self,k,dim="nrungs"):
        """
        Compute matrix representation of the squared supercurrent through the JJ in well k
        """
        if dim=="nrungs":
            dim = self.nrungs    
        
        cos_2phi = self.get_exp_alpha_phi_operator(2j, k,dim=dim) + self.get_exp_alpha_phi_operator(-2j, k,dim=dim)
        return self.J/4*(2*eye(dim)-cos_2phi)
        

    def compute_sqsc(self,dim="nrungs"):
        """
        Compute squared supercurrent through the JJ as block-diagonal matrix in full space.
        """
        sqsclist = zeros((self.nwells,self.nrungs,self.nrungs),dtype=complex)
                
        for k in range(0,self.nwells):
            sqsclist[k] = self.compute_sqsc_in_well(k-self.k0,dim)
            
        return sqsclist


    def compute_H_w(self,k,dim="nrungs"):
        """
        Compute matrix representation of Hamltonian H = H_LC + H_JJ in well k

        Parameters
        ----------
        k : int
            Well index requested
        dim : int, optional
            Number of rungs included. The default is self.nrungs.

        Returns
        -------
        H : ndarray((dim,dim),complex)
            Matrix representation of H in well k.

        """
        if dim=="nrungs":
            dim = self.nrungs 
            
        H_LC = self.compute_H_LC_w(k,dim=dim)
        H_JJ = self.compute_H_JJ_w(k,dim=dim)
        
        H = H_LC + H_JJ 
        
        H = array(H)
        return H
    

    def compute_H_wl(self):
        """
        Compute Hamiltonian as block-diagonal matrix in full space.
        
        Returns
        -------
        H_wl : ndarray((nwells,nrungs,nrungs),complex)
            Matrix representation of H, such that H_wl[k,a,b] = <k,a|H|k,b> 
        """
                
        Hlist = zeros((self.nwells,self.nrungs,self.nrungs),dtype=complex)
                
        for k in range(0,self.nwells):
            # print(f"  at well {k}/{self.nwells}")
            
            Hlist[k] = self.compute_H_w(k-self.k0)
            
            
        return Hlist
    

    def get_H_w(self,k):
        """
        Get matrix representation of Hamltonian H = H_LC + H_JJ in well k

        Parameters
        ----------
        k : int
            Well index requested

        Returns
        -------
        H : self.Hlist[k+self.k0]

        """

        H_w = self.H_wl[k+self.k0]
        return H_w
    

    def get_well_eigenfunction(self,n):
        """
        Get wavefunction of nth rung, as a function of phi - phi_n

        Parameters
        ----------
        n : int
            index of rung.

        Returns
        -------
        f : callable
            wavefunction of eigenstate.

        """

        f = get_ho_wavefunction(n,sigma=self.sigma )
        
        return f 
    

    def get_wavefunction(self,psi,max_rung="nrungs"):
        """
        Get wavefunction from matrix psi representing state in grid basis

        Parameters
        ----------
        psi : ndarray of complex or floats, (self.nwells,self.nrungs)
            state of system in grid basis (as a matrix.

        Returns
        -------
        f : callable
            wavefunction \psi(phi).

        """
        
        wf0 = self.G.get_wavefunction(psi,max_rung=max_rung)
        def wf(phi): 
            out = sqrt(1/self.sigma)*wf0(phi/self.sigma)
                
            return out 
        return wf  
    

    def compute_sz_wl(self):
        """
        Compute operator sgn(cos(phi/2)) (used to compute sigma_z) as wl object

        Returns
        -------
        SZ_wl : ndarray((nwells,nrungs,nrungs),dtype=complex)
            SZ_wl[k] gives the representation of cos(phi/2) in well basis 

        """
        SZ_wl = zeros((self.nwells,self.nrungs,self.nrungs),dtype=complex)
            
        # Construct sgn(cos(phi/2)) as a Fourier Series
        # The Fourier series is even in phi, so only has cosine harmonics of phi/2   
        for k in range(0,self.nwells):
            exp_phi_half = self.G.get_exp_alpha_x_operator(0.5j*self.sigma, k,dim=self.nrungs+self.step_function_order)
            Mat = exp_phi_half
            
            for n in range(0,self.step_function_order):
                cos_phi_half      = 0.5*(Mat  + Mat.conj().T)
                SZ_wl[k] += (4/pi)*cos_phi_half[:self.nrungs,:self.nrungs]/(2*n+1)*(-1)**n

                Mat = Mat @ exp_phi_half2
                
        return SZ_wl


    def get_S2_expval(self,psi):
        """
        Compute expectation value of stabilizer 2 for a state psi, expressed in 
        grid basis
    
        Parameters
        ----------
        psi : ndarray(nwells,nrungs)
            Initial state, as a matrix in grid basis. psi0_eb[m,k] gives the amplitude
            of the initial state in the kth harmonic of well m.
    
        Returns
        -------
        EV : float
            expectation value of stabilizer 2.
    
        """
        
        #Stabilizer 2 captures the overlap of the state with itself shifted by nu wells
        return real(self.G.ip(psi,roll(psi,self.nu,axis=0))) 
        

    def get_upper_rung_weight(self,psi,frac=.3):
        """
        Compute the total weight in the uppermost frac of rungs in each well.
        """
        n0 = int(self.nrungs*frac)
        return sum(abs(psi[:,-n0:])**2)/sum(abs(psi)**2)


    def get_outer_well_weight(self,psi,frac=.2):
        """
        Compute the total weight in the outermost frac of wells, for each rung.
        """
        n0 = int(self.nwells*frac/2)
        return (1-sum(abs(psi[n0:-n0,:])**2)/sum(abs(psi)**2))


    def get_S1_expval(self,psi):
        """
        Get expectation value of stabilizer 1, from the state psi in grid basis
    
        Parameters
        ----------
        psi : ndarray(nwells,nrungs)
            Initial state, as a matrix in grid basis. psi0_eb[m,k] gives the amplitude
            of the initial state in the kth harmonic of well m.
    
        Returns
        -------
        EV : float
            expectation value of stabilizer 1.
    
        """
        S1_psi = einsum("abc,ac->ab",self.S1_wl,psi)
        return real(self.G.ip(psi,S1_psi))
        

    def get_sqsc_expval(self,psi):
        """
        Get expectation value of squared supercurrent, from the state psi in grid basis
    
        Parameters
        ----------
        psi : ndarray(nwells,nrungs)
            Initial state, as a matrix in grid basis. psi0_eb[m,k] gives the amplitude
            of the initial state in the kth harmonic of well m.
    
        Returns
        -------
        EV : float
            expectation value of squared supercurrent.
    
        """
        sqsc_psi = einsum("abc,ac->ab",self.sqsc,psi)
        return real(self.G.ip(psi,sqsc_psi))
        

    def get_sz_expval(self,psi):
        """
        Get expectation value of sigma_z, from the state psi in grid basis
    
        Parameters
        ----------
        psi : ndarray(nwells,nrungs)
            Initial state, as a matrix in grid basis. psi0_eb[m,k] gives the amplitude
            of the initial state in the kth harmonic of well m.
    
        Returns
        -------
        EV : float
            expectation value of squared supercurrent.
    
        """
        SZ_psi = einsum("abc,ac->ab",self.sz_wl,psi)
        return real(self.G.ip(psi,SZ_psi))


    # def get_sx_expval(self,psi):
    #     sx_psi   = roll(psi,1,axis=0)
    #     V_sx_psi = self.G.apply_v(sx_psi)
    #     EV       = real(self.G.ip(psi,sx_psi))*exp(1/(2*self.sigma**2))
        
    #     return EV 
    
    # def get_sy_expval(self,psi):
    #     sx_psi    = roll(psi,1,axis=0)
    #     sz_sx_psi = einsum("abc,ac->ab",self.sz_wl,sx_psi)
    #     sy_psi    = 1j*sz_sx_psi 
        
    #     EV       = real(self.G.ip(psi,sy_psi))*exp(1/(2*self.sigma**2))
                        
    #     return EV 
    
    
    # def get_pauli_expvals(self,psi):
    #     SX = self.get_sx_expval(psi)
    #     SY = self.get_sy_expval(psi)
    #     SZ = self.get_sz_expval(psi)
        
    #     return array([SX,SY,SZ])


    def get_rung_weights(self,psi):
        """
        Return the weight of the wavefunction in each rung, summed over all wells
        """
        return sum(abs(psi)**2,axis=0)


    def get_well_weights(self,psi):
        """
        Return the weight of the wavefunction in each well, summed over all rungs
        """
        return sum(abs(psi)**2,axis=1)

    