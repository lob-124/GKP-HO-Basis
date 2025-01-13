#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 13:59:13 2023

Module for representing continuum variables with an overcomplete basis 
that consists of a grid of HO eigenstates.
@author: qxn582
	
	ifft(psi)[n] = 1/nwells \sum_{l=0}^nwells e^{2*pi*i*n*l/L}psi[l]
	fft(psi)[n]  =          \sum_{l=0}^nwells e^{-2*pi*i*n*l/L}psi[l]


	a[n] = e^{-2pi i n k/L} => ifft(a)[k] = 1 (while ifft(a)[l]=0 for l \neq k)
	
		=>
		
	ifft(psi)[n] = 1/nwells \sum_{l=0}^nwells e^{2*pi*i*n*l/L}psi[l]
	

Writing  

	psi[n] = \sum_{k=0}^nwells e^{-2*pi*i*k*n/L} psi_f[k]
	
Thus yields 

	ifft(psi)[n] = psi_f[n]
	
	
	
	now, (V @@ psi)[n] = \sum_{k=0}^nwells e^{-2*pi*i*k*n/L} [V(k)@ psi_f[k]]
	
""" 

from basic             import *
from units             import *
from numpy             import *
from numpy.linalg      import *
from matplotlib.pyplot import *

from numpy.random      import randint, rand
from numpy.fft         import fft,ifft
from gaussian_bath     import bath, get_J_ohmic

from numba import njit

THRESHOLD = 1e-3


@njit
def apply_k_array(karray,psi):
	"""
	Apply matrix in kspace
	"""
 
	fftpsi = fft(psi,axis=0)
	#Y      = einsum("abc,ac->ab",karray,fftpsi)
	Y = zeros((karray.shape[0],karray.shape[1]),dtype=complex128)
	for i in range(karray.shape[0]):
		Y[i,:] = karray[i] @ fftpsi[i]
	out    = ifft(Y,axis=0)
	
	return out 

@njit
def inner_prod(psi1,V_k,psi2):
	Vpsi2 = apply_k_array(V_k,psi2)
	return sum(psi1.conj()*Vpsi2)



class ho_grid():
	"""
	A class for representing quantum dynamics in grid of harmonic oscillator eigenstates. 
	The states have unit standard deviation, and spacing x_0.    
	
	More precisely, we construct a basis |N,n> where:
		<x|N,x> = \\frac{1}{\\sqrt{2^n*n!*\\sqrt{\\pi}}*e^{-(x-N*x_0)**2/2}*H_n(x-N*x_0)

	with H_n(x) the physicists Hermite polynomial. 
	
	"""
	def __init__(self,nwells,nrungs,grid_spacing,n_gauge_optimizers=1):
		self.nwells = nwells
		self.nrungs = nrungs 
	
		self.grid_spacing = grid_spacing
		
		self.overlap_matrices = self.compute_overlap_matrices()
		
		# Index of overlap matrix V^0_{ab} such that self.overlap_matrices[self.n0][a,b] = V^0_{ab}
		self.n0 = (shape(self.overlap_matrices)[0]-1)//2 
		
		# Index of central well
		self.k0 = self.nwells//2 
		
		self.overlap_matrices_k = self.get_Vk_array()
		self.V_inv_k,self.gauge_optimizer_k = self.compute_gauge_optimizer_and_S_matrix_inverter()
	

	#### ****
	####     Inter and intra well operators
	#### ****

	def get_x_operator_in_well_basis(self,k,dim="default",format="array"):
			"""
			Compute the operator X, represented in the basis of well k.

			We construct X as:
				X = k*x_0 + (a + a.conj().T)/sqrt(2)
			where a is the annihilation operator in well k


			Parameters
			----------
			k : int
			   index of well, for which we use eigenstates.
			dim : int, optional
			   Dimension of the operator. The default is self.nrungs.

			Returns
			-------
			X : sp.csc(dim,dim,dtype=complex)
			   Operator representing X in the eigenbasis of well <well_number>.

			"""
		   
			if dim=="default":
			   dim = self.nrungs
		   
			# Get the annihilation operator in this well, and create X
			well_annihilation_operator = get_annihilation_operator(dim,format=format)
			X = (k)*sp.eye(dim)*self.grid_spacing + 1/sqrt(2)*(well_annihilation_operator+well_annihilation_operator.T)
		   
			# Convert to dense matrix, if desired
			if format=="array":
			   X = array(X)
			return X


	def get_ddx_operator_in_well_basis(self,dim="default",format="array"):
		"""
		Compute the operator \partial_x, represented in the well basis 
		The operator is independent of well index, which is why it is not needed
		as a parameter.

		We construct X as:
				ddx = (-a + a.conj().T)/sqrt(2)
			where a is the annihilation operator in well k

		Parameters
		----------
		dim : int, optional
			Dimension of the operator. The default is self.nrungs.

		Returns
		-------
		ddx : sp.csc(dim,dim,dtype=complex)
			Operator representing \partial_\phi in the eigenbasis of the well 

		"""
		
		if dim=="default":
			dim = self.nrungs           
			
		# Get the annihilation operator in this well, and create ddx
		well_annihilation_operator = get_annihilation_operator(dim,format=format)
		return -1/(sqrt(2))*(well_annihilation_operator-well_annihilation_operator.T)


	def get_exp_alpha_x_operator(self,alpha,k,dim="nrungs"):
		"""
		Compute matrix representing e^{alpha*x} in basis of well k. 

		Parameters
		----------
		k   : int. Index of well
		
		dim : int, optional
			Dimension of matrix . The default is self.nrungs.

		Returns
		-------
		Exp : ndarray(dim,dim,dtype=complex)
			Matrix represnting e^{alpha*x} .
		"""
		if dim=="nrungs":
			dim = self.nrungs 
		
		max_order = get_maximal_order_of_exponential(dim, alpha,sigma=1)
		X       = self.get_x_operator_in_well_basis(k,dim=dim+max_order)
		
		return expm(alpha*X)[:dim,:dim]    
		
	
	def get_exp_alpha_ddx_operator(self,alpha,dim="nrungs"):
		"""
		Compute matrix representing e^{alpha*d/dx}. 
		This operator is well-independent.

		Parameters
		----------
		alpha : complex
		
		dim : int, optional
			Dimension of matrix . The default is self.nrungs.

		Returns
		-------
		Exp : ndarray(dim,dim,dtype=complex)
			Matrix represnting e^{alpha*d/dx} .

		Move this to basic.
		"""
		if dim=="nrungs":
			dim = self.nrungs 
		
		max_order = get_maximal_order_of_exponential(dim, alpha,sigma=1)
		ddX       = self.get_ddx_operator_in_well_basis(dim=dim+max_order)
		
		return expm(alpha*ddX)[:dim,:dim]
	   

	   
	#### ****
	#### ****
	####     Overlap matrices
	#### ****
	#### ****

	def compute_overlap_matrix(self,k,dim="nrungs"):
		"""
		Compute overlap matrix 
		
		V_{ab}^(k) = <k+n0,a|n0,b> 
				   = \\int dx \\psi_m(x+k*x_0)*\\psi_n(x)
		
		where \\psi_m(x) is the dimensionless QHO eigenstate, and x_0 the grid spacing.        

		Parameters
		----------
		k : int
			displacement of wavefunction.
		dim : int, optional
			Number of rungs included The default is self.nrungs 

		Returns
		-------
		out : ndarray(dim,dim), float, 
			overlap matrix, such that out[a,b] = V_{ab}^{(k)}

		"""
		if dim=="nrungs":
			dim = self.nrungs 
	
		if (self.grid_spacing*k/2) > ((sqrt(2*dim)*exp(1))):
			# Implement a cutoff beyond which the overlap between distant wells is zero
			Out =  zeros((dim,dim))
		else:
			# Construct the operator exp(-k*ddx*x_0) translating by k wells.
			#	The matrix elements of this operator correpsond to the desired overlaps.
			max_order = get_maximal_order_of_exponential(dim,alpha=k*self.grid_spacing)
			D = max_order + dim  
			
			ddx = self.get_ddx_operator_in_well_basis(dim=D,format="array")
			Out = expm(-k*ddx*self.grid_spacing)[:dim,:dim]
	  
		return Out  


	def compute_overlap_matrices(self,max_overlap_range = 100, threshold=1e-12):
		""" 
		Compute and return list of overlap matrices {V^k_{ab}}_k (see get_overlap_matrix())
		""" 
		out = zeros((2*max_overlap_range+1,self.nrungs,self.nrungs))
		n0 = max_overlap_range
		
		for n in range(0,max_overlap_range):
			# Compute overlaps for n  and -n
			mat1 = self.compute_overlap_matrix(n)
			mat2 = self.compute_overlap_matrix(-n)
			
			if norm(mat1) > threshold:	
				out[n0+n]  = mat1 
				out[n0-n]  = mat2 
			else:
				# If the overlap matrix is within tolerance of zero, stop
				overlap_range = n-1
				break
	
		return out[n0-overlap_range:n0+overlap_range+1]   
		

	def get_Vk_array(self):
		"""
		Compute the Fourier Transform of the overlap matrix V, i.e.,
				Vk_array[n] = \\sum_m e^{-2*pi*i*n*m/nwells}V^m
					= fft(Varray,axis=0)[n]
		
		We can also invert to obtain:
				V^n = \\sum_{k}e^{2*pi*i*n*k/nwells}Vk_array[k]/nwells 
					= ifft(Vk_array,axis=0)[n]
		
		The advantage of working with the Fourier transform is that V is easy to invert in 
			Fourier space (where it is block-diagonal)
	
		Returns
		-------
		Vk_array : ndarray(nwells,nrungs,nrungs,dtype=complex)
	
		"""
		if len(self.overlap_matrices)<self.nwells:
			Varray = concatenate((self.overlap_matrices,zeros((self.nwells-len(self.overlap_matrices),self.nrungs,self.nrungs))))
		else:
			Varray = array(self.overlap_matrices)
			
		Varray = roll(Varray,-self.n0,axis=0)
		return fft(Varray,axis=0)
		 
	
	def get_overlap_matrices(self):
		""" 
		Return list of overlap matrices {V^k_{ab}}_k
		"""
		return self.overlap_matrices
		

	def apply_vinv(self,psi):
		"""
		Apply the inverse of the overlap matrix (V) to the state psi.
		We do so in fourier space (with repsect to the well index), where
			inverting V is significantly simpler
		"""
		out = apply_k_array(self.V_inv_k, psi)       
		return out


	def apply_v(self,psi):
		"""
		Apply the overlap matrix (V) to the state psi.
		We also do this in fourier space (with repsect to the well index), where
			matrix multiplication is faster (because V is block-diagonal)
		"""
		out = apply_k_array(self.overlap_matrices_k,psi)
		return out     





	#### ****
	#### ****
	####     Gauge Optimization matrices
	#### ****
	#### ****

	def compute_gauge_optimizer_and_S_matrix_inverter(self,eta=1e-7):
		"""
		Compute operators that invert the S (overlap) matrix, and optimize the gauge.
		The operators are returned in Fourier space.
		"""

		# Define cost function for optimization problem
		# In essence, we penalize exponentially for having states with a large rung index         
		I0 = diag(exp(arange(self.nrungs)-(self.nrungs-1)) + eta*arange(self.nrungs)**2)

		# Output matrices
		# M: inverter 
		# Z: Gauge optimizers
		Marray_k = zeros((self.nwells,self.nrungs,self.nrungs),dtype=complex)
		Zarray_k = zeros((self.nwells,self.nrungs,self.nrungs),dtype=complex)
 
		
		evals_V_list = zeros((self.nwells,self.nrungs),dtype=complex)
		evals_Z_list = zeros((self.nwells,self.nrungs),dtype=complex)
		U_list = zeros((self.nwells,self.nrungs,self.nrungs),dtype=complex)
		U_inv_list = zeros((self.nwells,self.nrungs,self.nrungs),dtype=complex)
		for k in range(0,self.nwells):
			V = self.overlap_matrices_k[k]
			
			M_0 = V @ V + I0
			Z_0 = I0 + V
			Z = eye(self.nrungs) - inv(Z_0) @ I0

			# Diagonalize V and Z, and sort their spectra
			[evals_V,evecs_V] = eig(V)
			[evals_Z,evecs_Z] = eig(Z)
			argsorted_evals_Z = argsort(real(evals_Z))
			argsorted_evals_V = argsort(real(evals_V))
			evals_V_sorted = evals_V[argsorted_evals_V]
			evecs_V_sorted = evecs_V[:,argsorted_evals_V]
			evals_Z_sorted = evals_Z[argsorted_evals_Z]
			evecs_Z_sorted = evecs_Z[:,argsorted_evals_Z]

			# Store the change-of-basis matrices to the eigenbasis of Z, as well as 
			#	the eigenvalues
			U_list[k] = evecs_Z_sorted
			U_inv_list[k] = inv(evecs_Z_sorted) 
			evals_V_list[k] = evals_V_sorted
			evals_Z_list[k] = evals_Z_sorted
			
			# Compute M in this well
			M = inv(M_0) @ V
			Marray_k[k] = M


		Vmin = amin(evals_V_list,axis=0) 
		Vmax = amax(evals_V_list,axis=0)
		Zmin = amin(evals_Z_list,axis=0) 
		Zmax = amax(evals_Z_list,axis=0)
		n0 = argmax((Zmin[1:]-Zmax[:-1])*(Zmax[:-1]<1e-8))
		n0a = argmax((Vmin[1:]-Vmax[:-1])*(Vmax[:-1]<0.1))

		v0 = zeros((self.nrungs,),dtype=complex)
		v0[n0:]=1
		
		v0_a = zeros((self.nrungs,),dtype=complex)
		v0_a[n0a:] = 1
		
		P0 = diag(v0)
		P0_a = diag(v0_a)
		for k in range(0,self.nwells):
			Z = U_list[k] @ P0 @ U_inv_list[k]
			Zarray_k[k] = Z
		
		
		return Marray_k,Zarray_k 



	def optimize_gauge(self,psi):
		"""
		Optimize the gauge of the state psi, by applying the gauge optimize matrix.
		"""
		out =  apply_k_array(self.gauge_optimizer_k,psi)
		return out


	def get_well_eigenfunction(self,n):
		"""
		Get wavefunction of kth excited state of the dimensionless harmonic oscillator.
		   
			psi_n(x) = \frac{1}{\sqrt{2^n*n!*\sqrt{\pi}}*e^{-x**2/2}*H_n(x)

		with H_n(x) the physicists Hermite polynomial. 
		
		Parameters
		----------
		n : int
			index of rung requested

		Returns
		-------
		psi : callable
			wavefunction of the kth eigenstate.

		"""
		psi = get_ho_wavefunction(n)
		
		return psi 
	
	def get_wavefunction(self,psi,max_rung = "nrungs"):
		"""
		Get wavefunction from matrix psi representing state in grid basis

		Parameters
		----------
		psi : ndarray of complex or floats, (self.nwells,self.nrungs), or flattened 
			state of system in grid basis (as a matrix.

		Returns
		-------
		wf : callable
			wavefunction \psi(x) = \sum_{mn} psi[m,n]\psi_n(x-(n*self.grid_spacing)).

		"""
		if max_rung=="nrungs" or max_rung>=self.nrungs:
			max_rung = self.nrungs 
		sh = shape(psi)
		if sh == (self.nwells,self.nrungs):
			pass
		elif prod(sh)==self.nwells*self.nrungs:
			psi = psi.reshape((self.nwells,self.nrungs))
		sig = 1#*self.sigma 
		ws  = 1*self.grid_spacing
		
		nwells = shape(psi)[0]
		n0     = (nwells-1)//2 
		nr = 1*self.nrungs

		well_list = arange(nwells)-n0
		def wf(x):
			
			xmat = array([x - (k*ws) for k in well_list])      
			self.xmat = xmat
			out = zeros(shape(x),dtype=complex)
			for n in range(0,max_rung):
				f = get_ho_wavefunction(n)
				self.f = f                
				vec = nan_to_num(f(xmat))
				out += psi[:,n].T@vec

				
			return out 
		return wf 
	
	
	

	
	def apply_b_array(self,B,psi):
		"""
		apply Block array to psi, such that 
		
		Bpsi[n] = \sum_k Bmat[k]@psi[n-k]
		
		Assuming that Bmat[n] = B[n+len(Bmat)//2] 
		
		also requiring that len(Bmat)%2 = 1 
		
		"""
		
		assert(len(B)%2==1,"Length of B must be odd")
		
		n0 = len(B)//2 
		
		  
		psi = psi.T 
		Bpsi = 0*psi
		for n in range(-n0,n0+1):
				
			Bpsi +=    B[n+n0]@roll(psi,n,axis=1)
			
		Bpsi = Bpsi.T 
		
		return Bpsi 
		
		
	def apply_overlap_matrix(self,psi):
		"""
		apply overlap matrix to psi, such that 
		
		Vpsi[n,:] = \sum_k V[S.n0+k,:,:]@psi[n-k,:]
		
		(V\psi)_{n,a} = \sum_{k,b} V^{k}_{a,b} \psi_{n-k,b}
	
		NB: Using periodic boundary conditions
		
		Parameters
		----------
		psi : ndarray((nwells,nrungs))
			input state .
	
		Returns
		-------
		Vpsi : ndarray((nwells,nrungs))
			output state.
	
		"""
		
		psi = psi.T 
		Vpsi = 0*psi
		for n in range(-self.n0,self.n0+1):
				
			Vpsi +=    self.overlap_matrices[n+self.n0]@roll(psi,n,axis=1)#[:,:-n]
			
		Vpsi = Vpsi.T 
		
		return Vpsi
	   
	
	
	def apply_diag_array(self,diag_array,psi):
		"""
		Apply block-diagonal matrix to psi, such that 

		out[n] = diag_array[n] @ psi[n]
		
		Parameters
		----------
		diag_array : ndarray((self.nwells,self.nrungs,self.nrungs)), complex
		psi : ndarray((self.nwells,self.nrungs)), complex

		Returns
		-------
		out :  ndarray((self.nwells,self.nrungs)), complex

		"""
		assert len(diag_array)==self.nwells
		assert shape(psi)==(self.nwells,self.nrungs)
		
		out = einsum("abc,ac->ab",diag_array,psi)
		
		return out 

	def dot(self,a,b):
		"""
		Compute do product of operators A and B. Assuming 

		Parameters
		----------
		A : ndarray(nwells,d1,nrungs)
			matrix A.
		B : ndarray(nwells,nrungs,d2)
			matrix B.

		Returns
		-------
		C : ndarray(nwells,d1,d2).
		A * B
		"""
		
		# global A,B,C1,C2,C
		
		A = fft(a,axis=0)
		B = fft(b,axis=0)
		A = roll(A[::-1],1,axis=0)
		Vk  = self.overlap_matrices_k
		
		C1  = einsum("abc,acd->abd",Vk,B)
		C2  = einsum("abc,acd->abd",A,C1)
		C   = ifft(C2,axis=0)
		return C 
  
	def hc(self,mat):
		""" 
		Do hermitian conjugate in block representation
		""" 
		
		return mat.conj().swapaxes(1,2)
	

	def ip(self,psi1,psi2):
		"""
		Inner product of psi1 and psi2
		""" 
		Vpsi2 = self.apply_v(psi2)
		return sum(psi1.conj()*Vpsi2)
	

	def norm(self,v):
		"""
		Compute innner product of vectors v1,v2 

		Parameters
		----------
		v1 : ndarray(nwells,d1,nrungs)
			matrix A.
		v2 : ndarray(nwells,nrungs,d2)
			matrix B.

		Returns
		-------
		C : ndarray(nwells,d1,d2).
		A * B
		"""
		return real(sqrt(inner_prod(v,self.overlap_matrices_k,v)))