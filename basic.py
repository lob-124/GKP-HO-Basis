#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 10:36:07 2020

@author: frederik
edited by Liam
"""

from numpy import *
from scipy import *
from scipy.integrate import *
from scipy.linalg import *
import time
import os 
import sys
import datetime as datetime 
from scipy.special import factorial as factorial
import scipy.sparse as sp 
from scipy.special import hermite 

SX = array(([[0,1],[1,0]]),dtype=complex)
SY = array(([[0,-1j],[1j,0]]),dtype=complex)
SZ = array(([[1,0],[0,-1]]),dtype=complex)
I2 = array(([[1,0],[0,1]]),dtype=complex)

def hc(mat):
    return mat.conj().T


def get_ho_wavefunction(n,sigma=1):
    """
    Return dimensionless QHO wavefunction for the nth eigenstate, with vacuum
        fluctuation length sigma.
    """
    Ha = hermite(n)
    
    def f(phi):
        out =  (2**n*factorial(n)*sigma*sqrt(pi))**(-1/2)*exp(-phi**2/(2*sigma**2))*Ha(phi/sigma)
        return out 

    return f

    
def get_annihilation_operator(N,dtype=float,format="array"):
    """
    Return a matrix representation of the annihilation operator, in the basis of QHO
        eigenstates, truncated at eigenstate N.
    """
    global out
    
    if format=="array":
        out = sqrt(diag(arange(N)+1))
        
        if dtype==complex:
            out = out.astype(complex)

        out = concatenate((zeros((1,N)),out[:-1,:]))
        out = out.T
        return out
    else:
        out = sqrt(sp.diags(arange(N)+1,format=format))

        if dtype==complex:
            out = out.astype(complex)
        
        out = sp.vstack((zeros((1,N)),out[:-1,:]))
        out = out.T
        return out


def get_tmat(D,dtype=float,type="full"):
    """
    Return the translation matrix T, where 
                T[i,i+1]=1 , T[i,j]=0 for all other j
    Used in implementing the derivative (<-> conjugate momentum) operator
    """
    
    if type=="full":
        
        Tmat = eye(D,dtype=dtype)
        Tmat = roll(Tmat,1,axis=0)
        Tmat[0,-1] = 0
        Tmat[-1,0] = 0
    else:
        Tmat = sp.eye(D,format=type)
        return Tmat
        Tmat = sp.hstack(Tmat[1:,:],Tmat[-1:,:])
        
        Tmat[0,-1] = 0
        Tmat[-1,0] = 0
        
    return Tmat


def get_bloch_vector(rho):
    """
    Get bloch vector of 2x2 density matrix
    """
    out = [trace(x@rho) for x in (0.5*SX,0.5*SY,0.5*SZ)]
    return array(out)

    
def mod_center(x,y):
    """
    Return x mod y, with output in interval [-y/2,y/2)
    """
    mod_center = mod(x+y/2,y)-y/2
    return mod_center


def get_t_matrix(dim,offset=1,pbs=0):
    """
    Return a generalized translation matrix T, where 
                T[i,i+offset]=1 , T[i,j]=0 for all other j
    """
    M = eye(dim)
    M  = roll(M,offset,axis=1)
    
    if not pbs:
        #Account for open boundary conditions (if desired)
        if offset>0:  
            M[:,:offset]=0
        else:
            M[:,offset:]=0
    return M 


def binom(n,k):
    """
    returns n choose k
    """
    return int(prod([x for x in range(n-k+1,n+1)])/factorial(k,exact=1)+0.1)


def get_maximal_order_of_exponential(k,alpha,sigma=1):
    """
    Compute the order at which the taylor series of 
        e^(alpha x)|k> = e^(alpha*sigma (b+b^\dagger)/sqrt(2))|k> 
    converges, with |k> the kth HO eigenstate with vacuum fluctuation length sigma.  
    """
    n = abs(alpha)*sigma*(sqrt(2*k)*exp(1)+20)
    return int(n+1)
