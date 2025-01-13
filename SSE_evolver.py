#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 10:44:04 2023

@author: qxn582

Module for solving SSE

"""


from basic import *
from units import *
from numpy import *
from numpy.linalg import *
from matplotlib.pyplot import *
from numpy.random import rand
import scipy.sparse as sp
import scipy.sparse.linalg as spla 
from numpy.random import randint 
from numpy.random import default_rng

from time import perf_counter

class sse_evolver():
    """
    Object for evolving with SSE. 
    Takes as input a Hamiltonian, a jump operator, or a list of jump operator, and a timestep.
    
    Can be used to evolve state with the SSE over a duration <timestep>, using self.sse_evolve(psi)
    
    The Hamiltonian/jump operator can be a matrix, a callable (for time-dependent problems, ** this is not implemented yet**), 
    or an arbitrary object -- the latter case allows to exploit block-diagonal structures etc. 
    
    If the Hamiltonian and jump operators are not matrices, methods for matrix multiplication, norm, hermitian conjugation, and matrix exponentiation have to be specified, along with the identity operator. 
    Also takes as keyword argument resolution_order. The time-evolution over time_step is divided into 2**resolution_order equal steps.
    
    The solution implements the SSE with an efficient log-search routine that allows for very high time-resolution of the SSE at little (logarithmic) cost. 
    
    At this point, the method can only evolve a single vector at a time. 
    
    At this point, time-dependent Hamiltonians and jump oeprators are not implemented yet either
    """
    
    def __init__(self,Hamiltonian,Jump_operators,timestep,resolution_order=10,dot_method_matrix=dot,dot_method_vector=dot,norm_method = norm,hc_method = hc,identity = "standard",expm_method=expm,seed=None):
        if type(Hamiltonian) == callable:
            raise NotImplementedError("Time-dependent systems not implemented yet")
        else:
            self.mode = "static"
            
        if len(shape(Hamiltonian))>2:
            assert(shape(identity)==shape(Hamiltonian))
            self.I = identity
        else:
            self.I = eye(shape(Hamiltonian)[0],dtype=complex) 
        
        if resolution_order<0 or not type(resolution_order==int):
            raise ValueError("resolution order must be nonnegative integer")

        # Allow for non-standard multiplication methods to exploit block-diagonal structures
        self.matrix_shape = shape(Hamiltonian)
        self.vector_shape = self.matrix_shape[:-2]+self.matrix_shape[-1:]
 
        self.dot_matrix = dot_method_matrix
        self.dot_vector = dot_method_vector
        self.norm = norm_method
        self.hc = hc_method 
        self.expm = expm_method 
        
        self.H  = Hamiltonian
        
        if len(shape(Jump_operators))>2:
            self.nL  = shape(Jump_operators)[0]
        else:
            self.nL = 1
            Jump_operators = Jump_operators.reshape((1,)+shape(Jump_operators))
        
        self.Jump_operators = Jump_operators
        self.Heff = self.get_Heff()
        
        self.timestep = timestep 
        self.resolution_order = resolution_order
        self.dt0 = self.timestep/(2**resolution_order)
        
        self.Glist = self.get_Glist()

        self.rng = default_rng(seed)
        self.R = self.rng.random()
        

    def get_L(self,n):
        """
        Get the nth jump operator 
        """
        return self.L_list[n]


    def get_Heff(self):
        """
        Get effective Hamiltonian Heff = H - i/2 \\sum_k L^\\dagger_k @ L_k

        """
        Heff = self.H
        
        for n in range(0,self.nL):
            L= self.Jump_operators[n]
            Heff = Heff -0.5j*self.dot_matrix(self.hc(L),L)
            
        return Heff 
    
   
    def get_Glist(self):
        """
        Construct list of discrete time-evolution generators used for log-search routine
        
        Returns
        -------
        Glist : ndarray((self.resolution_order,)+self.matrix_shape)
            where
                Glist[n] = exp(-1j*dt_n * Heff), 
            and
                dt_n = self.time_increment*2^{-n}
        
        """
        Glist = zeros((self.resolution_order+1,)+self.matrix_shape,dtype=complex)
        dtlist   = self.timestep * 2.**(-arange(self.resolution_order+1))
        
        dG = self.expm(-1j*self.Heff*dtlist[-1]/hbar)
        for k in range(0,self.resolution_order+1):
            Glist[self.resolution_order-k] = dG
            if k==self.resolution_order:
                break 
            
            dG = self.dot_matrix(dG,dG)
            
        return Glist


    def generate_jump(self,psi_initial):
        """
        Generate quantum jump. 
        
        Apply a randomly selected jump operator to the state psi_initial, 
    
        Such that the relative probabilty of jump operator k (L_k) being applied is
        
        W_k = <psi|L^\\dagger_k L_k|\\psi>

        Parameters
        ----------
        psi_initial : initial state

        Returns
        -------
        psi_final : final state
        
        event : int
            index of jump operator that was applied.

        """
        outcome_list = zeros((self.nL,)+self.vector_shape,dtype=complex)
        
        for nj in range(0,self.nL):
            outcome_list[nj] = self.dot_vector(self.Jump_operators[nj],psi_initial)
            
        ### Identify the jump event    
        self.x = outcome_list
        
        # Create the probability mass function for the different events
        weightlist = ([self.norm(x)**2 for x in outcome_list])
        weightlist = weightlist/sum(weightlist)
        cweights = cumsum(weightlist)
        
        # Sample from the pmf above
        random_number = self.rng.random()
        event_ind  = amin(where(cweights > random_number)[0])
        psi_final = outcome_list[event_ind]   
        psi_final = psi_final/self.norm(psi_final)
        
        return psi_final,event_ind
        
      
    def sse_evolve(self,psi_initial):
        """
        Evolve psi by time self.time_step with SSE evolution
    
        Parameters
        ----------
        psi_initial: ndarray(self.vector_shape)
            Initial state
    
        Returns
        -------
        psi_final : ndarray(nwells,nrungs) 
            Final state, as a matrix in eigenbasis. psi0_eb[m,k] gives the amplitude
            of the final state in the kth well eigenstate in well m.
    
        jumps     : ndarray((*,2)), float
            List of jump events. 
            
            jumps[n,0] gives the time of the nth jump
            
            jumps[n,1] gives the type of the nth jump , i.e., the nth jump was
                       was generaed by self.Jump_operators[s,1], with 
                       s =jumps[n,1]. 
                       
        Note that jumps = zeros((0,2)) if no jumps occured 
        """
        
        # Make sure psi is normalized before we begin 
        psi_initial = psi_initial/self.norm(psi_initial)
        
        # Sample a random number. We will monitor the norm square of psi to see when it 
        #   drops below R 
        R = self.rng.random()
        
        step_size_list = 2**(arange(self.resolution_order+1))[::-1]
        Nsteps         = 2**(self.resolution_order)
        
        psi = 1*psi_initial  
        s = 0 
        
        nit = -1
        jumplist = zeros((0,2))
        while s<Nsteps:
            
            # Loop through the logarithmically spaced evolution operators, applying each and checking
            #   the norm of the state against R
            for k in range(0,self.resolution_order+1):
                ds = step_size_list[k]
                if s+ds <= Nsteps: # Don't evolve for longer than self.time_step in TOTAL
                    G = self.Glist[k]
                    psi_new = self.dot_vector(G,psi)
                    new_weight = (self.norm(psi_new))**2
                    
                    if new_weight > R:
                        # If the norm has NOT dropped below R, then update the state to save application
                        #   of this evolution operator
                        s += ds
                        psi = psi_new 
                
            if s < Nsteps:
                # If the above loop terminates, and we haven't reach total evolution time self.time_step,
                #   then a jump occurred. Generate the result of the jump accordingly   
                X = self.generate_jump(psi)
                psi,event = X
                
                # Record the jump time and type
                jump_time = s/Nsteps
                jumplist =  concatenate((jumplist,array([[jump_time,event]])))

                # Re-sample R
                R = self.rng.random()
        
        #end while loop             

        # Re-normalize the state, and return
        psi = psi / self.norm(psi)
        return psi,jumplist