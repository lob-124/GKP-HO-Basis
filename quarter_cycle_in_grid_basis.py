#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 12:01:06 2023

@author: qxn582

Implementing quarter cycle in grid basis using the analytic formula from the notes:
    
    <m,j|U_0|n,k>   =   (-i)^{j-k+2mn}
                        \\frac{\\sigma^2}{\\sqrt{2} r }
                        \\sum_z\\frac{1}{z!}\\left(\frac{\\sigma^2}{4\\pi i r^2}\\right)^z
                        [T^z V]_{km}[T^z V]_{jn}
                        
    in 0 mode 
    
with 

    V_{kn}          = \\psi_{k}(n\\sigma^2/2r), 
    T_{jk}          = \\sqrt{k/2}\\delta_{j,k-1}
                     -\\sqrt{(k+1)Ø/2}\\delta_{j,k+1}
                     
The code returns the tensor M, where

    M[m,j,n,k]      = <m,j|U_0|n,k>  
    
"""

from basic import *
from numpy import *
from numpy.linalg import *
from matplotlib.pyplot import *
from scipy.special import factorial,hermite
from units import *
from gaussian_bath import bath,get_J_ohmic
from basic import tic,toc
from scipy.interpolate import interp1d
from numpy.random import rand
import scipy.sparse as sp
import scipy.sparse.linalg as spla 
from scipy.special import hermite
from numpy.random import randint 
from numpy.fft import fft
from numpy.random import rand

#A custom library computing the QHO wavefunctions with precision math to avoid overflow errors
from precision_math import *     
    


def find_max_order(ratio,nrungs):
    """
    Find max order where infinite sum in (1) below is truncated
    V and T are calculated with dimension <max_order> + nrungs to avoid issues with truncating 
        in rung space
    """
    z= 0 
    while True :
        x = log(nrungs+z)*nrungs+log(abs(ratio))*z  
        if x<log(1e-14):
            break
        
        z += 1
        
    return z + 3


def get_V_matrix(sigma,r,n0,nrungs,max_order):
    """
    Calculate V-matrix 
                V[k,n] = \\psi_{k}(n\\sigma^2/2r), 
    with \\psi_k the kth HO-eigenstate with standard deviation sigma
    """
    phivec = arange(-n0,n0+1,dtype=float)*sigma**2 /(2*r)
    
    V = zeros((max_order+nrungs,nwells))
    for k in range(0,nrungs+max_order):
        f = my_ho_wavefunction(k,sigma=sigma)
        for i,phi in enumerate(phivec):
            V[k,i] = f(phi)
    
    return V 


def get_U0_tensor(T,sigma,r,nrungs,nwells,max_order):
    """
    Construct the U0 tensor, given by:
            U0[m,j,n,k]  = \\frac{ \\sigma^2}{\\sqrt{2 }r}\\sum_z\\frac{1}{z!}(\\frac{\\sigma^2}{4\\pi i r^2})^z
                                [T^z V]_{km}[T^z V]_{jn}
    """
    # First construct tensor U, where <m,j|U|n,k> = (-i)^{j-k+2mn} U0[j,n,k,m]
    #
    ### Calculate U matrix 
    M = array(V)
    
    U0 = zeros((nrungs,nwells,nrungs,nwells),dtype=complex)
    # Construct U0 tensor iteratively
    for z in range(0,max_order):
        U0 += -1j**z * tensordot(M[:nrungs,:],M[:nrungs,:],axes=0)
        M = T @ M*sqrt(sigma**2/(4*pi*r**2)/(z+1))
           
        # Terminate if squared Frobenius norm of M is below floating-point accuracy
        M_norm = (norm(M)**2)
        if M_norm <1e-14:
            break

    # Swap axes of U0, such that U0[k,m,j,n] -> U0[m,j,n,k]
    U0 =  U0*sigma**2/(sqrt(2)*r)
    U0 =  U0.swapaxes(3,2)
    U0 =  U0.swapaxes(1,2)
    U0 =  U0.swapaxes(0,1)    

    return U0


def get_W_tensor(n0,nrungs,nwells):
        """
        Get the W tensor, which has components:
                W[m,j,n,k] =  (-i)^{j-k+2mn}
        """
        mvec = arange(-n0,n0+1)
        kvec = (-1j)**arange(0,nrungs)
        
        W_tensor = (-1.)**tensordot(mvec,mvec,axes=0).reshape((nwells,1,nwells,1))       
        K_tensor = tensordot(kvec,kvec.conj(),axes=0).reshape((1,nrungs,1,nrungs))
        
        return W_tensor*K_tensor



def get_quarter_cycle_matrix(grid_shape,sigma,r):
    """
    Calculate matrix elements of quarter cycle evolution in grid basis

    Parameters
    ----------
    grid_shape = (nwells,nrungs): (int,int)
        Shape of grid. We work in 0 mode, so nwells should be odd
    sigma : float
        Standard deviation of well states.
    r : float
        well spacing, in units of 2pi
        
    Returns
    -------
    U : ndarray((nwells*nrungs,nwells*nrungs)), complex
    
        Matrix encoding the quarter cycle evolution. 
        Specifically, 
        
        <m,j|U_0|n,k>  =  U[nrungs*m+j,nrungs*n+k].


    Details 
    -------
    To compute U, we use the formula
    
        <m,j|U_0|n,k> = (-i)^{j-k+2mn} \\frac{ \\sigma^2}{\\sqrt{2 }r}
                        \\sum_z\\frac{1}{z!}\\left(\\frac{\\sigma^2}{4\\pi i r^2}\\right)^z
                        [T^z V]_{km}[T^z V]_{jn}.
        (1)

    
    where
    
        V_{kn} \\equiv \\psi_{k}(n\\sigma^2/2r)
        
        T_{jk} \\equiv \\sqrt{\frac{k}{2}}\\delta_{j,k-1}-\\sqrt{\\frac{k+1}{2}}\\delta_{j,k+1}
    
        (2)
 
    
    """
    nwells,nrungs = grid_shape    
    n0    = nwells//2 
    ratio = sigma**2/(4*pi*1j*r**2)
    
    assert abs(ratio)<1,"ratio is too large -- things will not converge"

    #Find the order at which to truncate the sums above, and compute the V matrix
    max_order = find_max_order(ratio,nrungs)
    V = get_V_matrix(n0,sigma,r,max_order,nrungs)
    
    # Compute the  T  matrix:
    #           T[j,k] = \sqrt{\frac{k}{2}}\delta_{j,k-1}-\sqrt{\frac{k+1}{2}}\delta_{j,k+1}
    B = get_annihilation_operator(max_order+nrungs)
    T = sqrt(0.5)*(B-B.T)
    
    #Compute the U0 tensor, which differs from U by a multiplication by the W tensor, which 
    #   has components   
    #               W[m,j,n,k] =  (-i)^{j-k+2mn}
    U0 = get_U0_tensor(T,sigma,r,nrungs,nwells,max_order)
    
    # Compute the aforementioned W tensor, and compute U
    W = get_W_tensor((n0,nrungs,nwells))
    U = U0*W
    
    # Reshape U to matrix 
    return U.reshape((nwells*nrungs,nwells*nrungs))
    