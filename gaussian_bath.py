#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 17:06:30 2021

@author: frederiknathan
Edited by Liam O'Brien

Moudule generating spectral functions and jump operators from gaussian baths. 
Core object is Bath.
"""

from matplotlib.pyplot import *
from numpy import *
import numpy.fft as fft
from numpy.linalg import *
import warnings
from basic import SX,SY,SZ
warnings.filterwarnings('ignore')
from scipy.interpolate import RectBivariateSpline
import sys

import numpy.random as npr 

RESOLUTION_FOR_INTERPOLATOR = 50
OUTPUT_STEPS = 100
def window_function(omega,E1,E2):
    """
    Return a Gaussian window function with center (E1+E2)/2, and width (E1-E2)
    """
    
    Esigma = 0.5*(E1-E2)
    Eav = 0.5*(E1+E2)
    X = exp(-(omega-Eav)**2/(2*Esigma**2))
    if sum(isnan(X))>0:
        raise ValueError
        
    return X
    

def S0_colored(omega,E1,E2,omega0=1):
    """
    Spectral density of colored noise (in our model)
    """
    A1 = window_function(omega, E1, E2)
    A2 = window_function(-omega, E1, E2)
    
    return (A1+A2)*abs(omega)/omega0


def get_ohmic_spectral_function(Lambda,omega0=1,symmetrized=True):
    """
    Generate spectral function

    S(\omega) = |\omega| * e^{-\omega^2/2\Lambda^2}/\omega_0
    
    Parameters
    ----------
    Lambda : float
        Cutoff frequency.
    omega0 : float, optional
        Normalization. The default is 1.
    symmetrized : bool, optional
        indicate if the spectral function shoud be symmetric or antisymmetric. If False, |\omega| -> \omega in the definition of S. 
        The default is True.

    Returns
    -------
    S : method
        spectral function S,

    """
    if symmetrized :
            
        def f(omega):
            return abs(omega)*exp(-(omega**2)/(2*Lambda**2))/omega0 
    else:
        def f(omega):
            return (omega)*exp(-(omega**2)/(2*Lambda**2))/omega0
        
    return f


def S0_ohmic(omega,Lambda,omega0=1):
    """
    Spectral density of ohmic bath
    
        S(omega) = |omega|*e^(-omega^2/(2*Lambda**2))
         
    """
    
    Out = abs(omega)*exp(-(omega**2)/(2*Lambda**2))/omega0
    
    return Out


def BE(omega,Temp=1):
    """
    Return Bose-Einstein distribution function at temperature Temp. 
    """
    return (1)/(1-exp(-omega/Temp))*sign(omega)


def get_J_colored(E1,E2,Temp,omega0=1):
    """
    generate spectral function of colored bath at given values of E0,E1,Temp 
    
    Returns spectral function as a function/method
    """
    dw = 1e-12
    nan_value = S0_colored(dw,E1,E2,omega0=omega0)*BE(dw,Temp=Temp)
    def J(omega):
        
        return nan_to_num(S0_colored(omega,E1,E2,omega0=omega0)*BE(omega,Temp=Temp),nan=nan_value)
    
    return J


def get_J_ohmic(Temp,Lambda,omega0=1):
    """
    generate spectral function of ohmic bath, modified with gausian as cutoff,
    
    S(omega) = |omega|*e^(-omega^2/(2*Lambda**2))

    """
    def J(omega):
        out  = nan_to_num(S0_ohmic(omega,Lambda,omega0=omega0)*BE(omega,Temp=Temp),nan=Temp/omega0)
        if len(shape(omega))>0:
            out[where(abs(omega)<1e-14)] = Temp/omega0
        return out 
    
    return J 


def get_J_from_S(S,temperature,zv):
    """
    Get bath spectral function from bare spectral function at a given temperature. 
    Zv specifies what value to give at zero (Where BE diverges)
    """
    def out(energy):
        return nan_to_num(BE(energy,Temp = temperature)*S(energy)*sign(energy),nan=zv)
    
    return out


def get_g(J):
    """
    Get jump spectral function from given bath spectral function, J(omega)
    """
    def g(omega):
        return sqrt(abs(J(omega))/(2*pi))
    
    return g


def get_ft_vector(f,cutoff,dw):
    """
    Return fourier transform of function as vector, with frequency cutoff <cutoff> and frequency
        resolution <dw>.
    Fourier transform: \int dw e^{-iwt} J(w)
    """ 
    omrange = linspace(-cutoff,cutoff,2*int(cutoff/dw)+1)[:-1]
    n_om  = len(omrange)
    omrange = concatenate((omrange[n_om//2:],omrange[:n_om//2]))
    
    vec    = fft.fft(f(omrange))*dw 
    times   = 2*pi*fft.fftfreq(n_om,d=dw)
    AS = argsort(times)
    times = times[AS]
    vec = vec[AS]
    
    return times,vec
        
        
        
class bath():
    """
    bath object. Takes as input a spectral function. Computes jump correlator 
    and ULE timescales automatically. 
    Can plot correlation functions and spectral functions as well as generate 
    jump operators and Lamb shfit
    
    Parameters
    ----------
        J : callable.     
            Spectral function of bath. Must be real-valued
        cutoff : float, >0.    
            Cutoff frequency used to compute time-domain functions (used to 
            compute Lamb shift and ULE timescales, and for plotting correlation 
            functions).
        dw : float, >0.  
            Frequency resolution to compute time-domain functions (see above)
        
    Properties
    ----------
        J : callable.  
            Spectral function of bath. Same as input variable J
        g : callable.  
            Fourier transform of jump correlator (sqrt of spectral function)
        cutoff : float.    
            Same as input variable cutoff
        dw : float.     
            Same as input variable dw
        dt : floeat.    
            Time resoution in time-domain functions. Given by pi/cutoff
        omrange : ndarray(NW)    
            Frequency array used as input for computation of time-domain 
            observables (see above). Frequencies are in range (-cutoff,cutoff)
            and evenly spaced by dw. Here NW is the length of the resulting 
            array.
        times : ndarray(NW)     
            times corresponding to time-domain functions
        correlation_function : ndarray(NW), complex
            Correlation function at times specified in times. 
            Defined such that correlation_function[z] = J(times[z]).
        jump_correlator  :ndarray(NW), complex    
            Jump correlator at times specified in times 
        Gamma0 : float, positive.    
            'bare' Gamma energy scale. The ULE Gamma energy scale is given by 
            gamma*||X||*Gamma0, where gamma and ||X|| are properties of the 
            system-bath coupling (see ULE paper), and not the bath itself. 
            I.e. gamma, ||X|| along with Gamma0 can be used to compute Gamma.
        tau : float, positive.      
            Correlation time of the bath, as defined in the ULE paper.
        

    """
    
    def __init__(self,J,cutoff,dw):
        self.J = J
        self.g = get_g(J)
        
        self.cutoff = cutoff
        self.dw     = dw
        self.dt     = 2*pi/(2*cutoff)
        
        # Range of frequencies 
        self.omrange = linspace(-cutoff,cutoff,2*int(cutoff/dw)+1)[:-1]+dw/2

        # Compute correlation function of bath
        self.times,self.correlation_function = self.get_ft_vector(self.J)
        Null,self.jump_correlator = self.get_ft_vector(self.g)
                
        g_int = sum(abs(self.jump_correlator))*self.dt 
        
        #Compute various constants describing bath
        self.K_vec  = cumsum(self.correlation_function[::-1])[::-1]*self.dt
        K_int  = sum(abs(self.K_vec[self.times>=0]))*self.dt
        self.lambda_const = 4*K_int
        self.Gamma0 = 4*g_int**2
        self.tau = sum(abs(self.jump_correlator*self.times))*self.dt/g_int 
        self.dephasing_speed = 4*pi*self.J(0)
        self.GammaJtau = 4*sum((self.times*abs(self.correlation_function))[self.times>=0])*self.dt

    
    def get_ft_vector(self,f) :
        """
        Return fourier transform of function as vector, with frequency cutoff <cutoff> and 
            frequency resolution <dw>.
        Fourier transform: \int dw e^{-iwt} J(w)
        """
        cutoff= self.cutoff
        dw    = self.dw
        omrange = linspace(-cutoff,cutoff,2*int(cutoff/dw)+1)[:-1]
        n_om  = len(omrange)
        omrange = concatenate((omrange[n_om//2:],omrange[:n_om//2]))
        
        #Do the FFT, and re-arrange the output
        vec    = fft.fft(f(omrange))*dw 
        times   = 2*pi*fft.fftfreq(n_om,d=dw)
        AS = argsort(times)
        times = times[AS]
        vec = vec[AS]
        
        return times,vec
    

    def get_ule_jump_operator(self,X,H,return_ed=False,gamma=1):
        """
        Get jump operator for bath, associated with operator X and Hamiltonian H
        (all must be arrays)      

        Parameters
        ----------
        X : ndarray
            The (system) operator(s) appearing in the system-bath coupling.
            If len(X.shape) == 3, the first axis indexes the coupling operator, and the
                other indices index the component of each operator.
        H : ndarray
            The system Hamiltonian.
        return_ed : bool, optional
            Whether or not to return the eigenvalues and eigenvectors of H. Defaults to False
        gamma: float, optional
            System-bath coupling strength - an overall scale factor multiplyin the jump operator.
                Defaults to 1.

        Returns
        -------
        L : ndarray
            The ULE jump operator, in the same basis as X and H
        """
        
        #Diagonalize the system Hamiltonian
        [E,V]=eigh(H)
        ND = len(E)

        #Compute the matrix M with components
        #       M_{ij} = g(E_j - E_i)
        Emat = outer(E,ones(ND))
        Emat = Emat.T-Emat
        self.Emat = Emat
        self.corr_of_diffs = self.g(Emat)   #The matrix M above
         
        if len(shape(X))==2:
            X_eb = V.conj().T @ X @ V   #Transform X to eigenbasis (eb) of H

            # Multiply (elementwise!) X_eb by the matrix M defined above
            L = 2*pi*V @ (X_eb *self.corr_of_diffs ) @ (V.conj().T)
            L_out = L * (abs(L)>1e-13)

        elif len(shape(X))==3:  #Handle the case when there are multiple operators
            Nop = shape(X)[0]
            
            L_out  = zeros(shape(X),dtype=complex)
            for nop in range(0,Nop):
                X_eb = V.conj().T @ X[nop] @ V  #Transform X_nop to eigenbasis (eb) of H

                # Multiply (elementwise!) X_eb by the matrix M defined above in each block
                L = 2*pi*V@(X_eb *self.corr_of_diffs )@(V.conj().T)
                L_out[nop] = L * (abs(L)>1e-13)
            
        else:
            DH = shape(H)[0]
            raise ValueError(f"X must be array of dimension ({DH,DH}) or (N,{DH,DH})")
            
        L_out = sqrt(gamma)*L_out
        if not return_ed:
            return L_out 
        else:
            return L_out,[E,V]
        
         
    def get_cpv(self,f,real_valued=True):
        """ 
        Return Cauchy principal value of integral \\int dw f(w)/(w-w0) 
        The integral is defined as:
                Re ( \\int dw f(w)Re ( 1/(w-w0-i0^+)))
        This is the same as 
                i/2 *  \\int_-\\infty^\\infty dt f(t)e^{-0^+ |t|} sgn(t)    
        where  
                f(t) =     \\int d\\omega f(\\omega)e^{-i\\omega t} 
        (i.e. get_time_domain_function(f))  
        """
        #Set up the integration domain and the integrand
        S0 = shape(f(0))
        nd = len(S0)
        Sw = (len(self.omrange),)+(1,)*(nd-1)
        wrange = self.omrange.reshape(Sw)
        vec1 = f(wrange)
        vec2 = f(-wrange)
        
        # Compute (f(omega) - f(-omega))/omega
        vec = 0.5*(vec1-vec2)/(wrange)
        dw = 1e-10
            
        # Handle very small omega separately
        if amin(abs(wrange))<1e-12:
            ind = where(abs(wrange)<1e-12)[0] 
            vec[ind] = 0.5*(f(dw)-f(-dw))/dw
            
        #Sum the result to obtain the value of the integral
        return sum(vec,axis=0)*self.dw 
 

    def get_lamb_shift_amplitudes(self,q1list,q2list):
        """
        Get amplitude of lamb shift F_{\alpha \beta }(q1,q2) (see L)
        
        q1 and q2 must be 1d arrays of the same length
        """
        nq = len(q1list)
        
        q1list = q1list.reshape(1,nq)
        q2list = q2list.reshape(1,nq)
        def f(x):
            return self.g(x-q1list)*self.g(x+q2list)
        
        return -2*pi*self.get_cpv(f)    
    

    def create_lamb_shift_amplitude_interpolator(self,cutoff,resolution):
        """
        Create an interpolator for the Lamb shift amplitudes, using get_lamb_shift_amplitudes()
            above.
        """
        assert(type(resolution)==int)
        
        # Define a grid of E values and compute the amplitudes on the grid
        Evec = linspace(-cutoff,cutoff,resolution)
        E1,E2 = meshgrid(Evec,Evec)
        amplitudes = self.get_lamb_shift_amplitudes(E2.flatten(),E1.flatten()).reshape(shape(E1))
        
        # Create interpolators for the real and imaginary parts
        interpolator_r = RectBivariateSpline(Evec,Evec,real(amplitudes))
        interpolator_i = RectBivariateSpline(Evec,Evec,imag(amplitudes))        
        return Evec,amplitudes,interpolator_r,interpolator_i
       

    def get_lambda0(self):
        """
        Compute \\Lambda_0 = \\mathcal P \\int dw J(w)/w
        """
        return self.get_cpv(self.J)
 
    
    def get_ule_lamb_shift_static(self,X,H):
        """
        Compute the ULE Lamb shift (LS) for a static Hamiltonian, using self.get_ft_vector to 
            calculate cauchy p.v.

        Parameters
        ----------
        X : ndarray
            The (system) operator(s) appearing in the system-bath coupling.
            If len(X.shape) == 3, the first axis indexes the coupling operator, and the
                other indices index the component of each operator.
        H : ndarray
            The system Hamiltonian.
        
        Returns
        -------
        L : ndarray
            The ULE jump operator, in the same basis as X and H
        """
             
        # Diagonalize the system Hamiltonian
        [E,U]=eigh(H) 
        D  = shape(H)[0]
    
        X_b = U.conj().T.dot(X).dot(U)  #Transform X to eigenbasis (eb) of H
        LS_b = zeros((D,D),dtype=complex)   #Initialize LS matrix
          
        # Generate the interpolator for the amplitudes
        Emin = amin(E)
        Emax = amax(E)
        cutoff = 1.05*(Emax-Emin)
        resolution = RESOLUTION_FOR_INTERPOLATOR
        Evec,Values,Q_r,Q_i = self.create_lamb_shift_amplitude_interpolator(cutoff, resolution)      
        
        # Populate the LS matrix, using the interpolators above to compute each matrix element
        for m in range(0,D):
            for n in range(0,D):
                
                E_mn = E[m]-E[n]
                E_nl_list = E[n]-E
                E_mn_list= E_mn*ones(len(E))
                Amplitudes = Q_r(E_mn_list,E_nl_list,grid=False)+1j*Q_i(E_mn_list,E_nl_list,grid=False)

                LS_b[m] += Amplitudes*X_b[m,n]*X_b[n,:]
        
        return U.dot(LS_b).dot(U.conj().T)

#end bath class