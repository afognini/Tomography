#!/usr/local/bin/python
# -*- coding: utf-8 -*-

#  Copyright (c) 2016-2017 Tjeerd Fokkens, Andreas Fognini, Val Zwiller
#  Licensed MIT: http://opensource.org/licenses/MIT

import numpy as np
import scipy.linalg
from scipy.optimize import minimize, rosen, rosen_der
from scipy.linalg import fractional_matrix_power
from multiprocessing import Pool
import time


class DensityMatrix(object):
	"""Computes the density matrix for an optical two qubit system.
	The measurements are performed with only one detector for each qubit.
	The code is programmed along the procedure described in D. F. V. James et al. Phys. Rev. A, 64, 052312 (2001).

	The measurements need to be performed in horizontal (H) or vertical (V),
	circular right (R) or left (L), and diagonal (D) or antidiagonal (A) projections.

	We use the following vector representation:
		* :math:`H=\\begin{pmatrix}1 \\\ 0 \end{pmatrix}`
		* :math:`V=\\begin{pmatrix}0 \\\ 1 \end{pmatrix}`
		* :math:`R=\\frac{1}{\sqrt{2}}\\begin{pmatrix}1 \\\ -i \end{pmatrix}`
		* :math:`L=\\frac{1}{\sqrt{2}}\\begin{pmatrix}0 \\\ i \end{pmatrix}`
		* :math:`D=\\frac{1}{\sqrt{2}}\\begin{pmatrix}1 \\\ 1 \end{pmatrix}`
		* :math:`A=\\frac{1}{\sqrt{2}}\\begin{pmatrix}1 \\\ -1 \end{pmatrix}`

	:param array basis: An array of basis elements in which the measurement is performed.
	"""
	def __init__(self, basis):
		"""Intialize the DensityMatrix class.
		The following vectors and matrices are initiad:
		:math:`\psi_\\nu` 16 basis vectors spanning qubit space and
		:math:`M, B, \Gamma`, and Pauli matrices are initiated.

		Args:
			basis
				An array of basis elements in which the measurement is performed.

		Returns:
			No return.

		Raises:
			ValueError
				If Gamma matrices are not defined properly.

		Example:

			.. code-block:: python

				basis =  ["HH","HV","VV","VH","HD","HR","VD","VR","DH","DV","DD","DR","RH","RV","RD","RR"]
		"""
		self.basis = basis

		self.H = np.array([1,0])
		self.V = np.array([0,1])
		self.R = 1/np.sqrt(2)*(self.H - 1j*self.V)
		self.L = 1/np.sqrt(2)*(self.H + 1j*self.V)
		self.D = 1/np.sqrt(2)*(self.H +    self.V)
		self.A = 1/np.sqrt(2)*(self.H -    self.V)

		#Generate psi_nu's and gamma's
		self.PSI = []
		self.GAMMA = []

		#Fill PSI
		for basis_element in self.basis:
			basis_branch1 = self.basis_str_to_object(basis_element[0])
			basis_branch2 = self.basis_str_to_object(basis_element[1])
			self.PSI.append(np.hstack(np.outer(basis_branch1,basis_branch2)))

		#Fill Gamma
		gam1  = (1/2.)*np.array([[0,1,0,0]  ,[1,0,0,0]  ,[0,0,0,1]  ,[0,0,1,0]])
		gam2  = (1/2.)*np.array([[0,-1j,0,0],[1j,0,0,0] ,[0,0,0,-1j],[0,0,1j,0]])
		gam3  = (1/2.)*np.array([[1,0,0,0]  ,[0,-1,0,0] ,[0,0,1,0]  ,[0,0,0,-1]])
		gam4  = (1/2.)*np.array([[0,0,1,0]  ,[0,0,0,1]  ,[1,0,0,0]  ,[0,1,0,0]])
		gam5  = (1/2.)*np.array([[0,0,0,1]  ,[0,0,1,0]  ,[0,1,0,0]  ,[1,0,0,0]])
		gam6  = (1/2.)*np.array([[0,0,0,-1j],[0,0,1j,0] ,[0,-1j,0,0],[1j,0,0,0]])
		gam7  = (1/2.)*np.array([[0,0,1,0]  ,[0,0,0,-1] ,[1,0,0,0]  ,[0,-1,0,0]])
		gam8  = (1/2.)*np.array([[0,0,-1j,0],[0,0,0,-1j],[1j,0,0,0] ,[0,1j,0,0]])
		gam9  = (1/2.)*np.array([[0,0,0,-1j],[0,0,-1j,0],[0,1j,0,0] ,[1j,0,0,0]])
		gam10 = (1/2.)*np.array([[0,0,0,-1] ,[0,0,1,0]  ,[0,1,0,0]  ,[-1,0,0,0]])
		gam11 = (1/2.)*np.array([[0,0,-1j,0],[0,0,0,1j] ,[1j,0,0,0] ,[0,-1j,0,0]])
		gam12 = (1/2.)*np.array([[1,0,0,0]  ,[0,1,0,0]  ,[0,0,-1,0] ,[0,0,0,-1]])
		gam13 = (1/2.)*np.array([[0,1,0,0]  ,[1,0,0,0]  ,[0,0,0,-1] ,[0,0,-1,0]])
		gam14 = (1/2.)*np.array([[0,-1j,0,0],[1j,0,0,0] ,[0,0,0,1j] ,[0,0,-1j,0]])
		gam15 = (1/2.)*np.array([[1,0,0,0]  ,[0,-1,0,0] ,[0,0,-1,0] ,[0,0,0,1]])
		gam16 = (1/2.)*np.array([[1,0,0,0]  ,[0,1,0,0]  ,[0,0,1,0]  ,[0,0,0,1]])

		self.GAMMA = np.array([gam1,gam2,gam3,gam4,gam5,gam6,gam7,gam8,gam9,gam10,gam11,gam12,gam13,gam14,gam15,gam16])

		sig_1 = np.complex_(np.array([[0, 1],
						  			[1, 0]]))
		sig_2 = np.complex_(np.array([[0,-1j],
						  			[1j,0]]))
		sig_3 = np.complex_(np.array([[1, 0],
						  			[0,-1]]))
		#Defining the Pauli matrices
		self.PAULI = np.array([sig_1, sig_2, sig_3])

		#Test orthogonalty of self.GAMMA
		if self.test_gamma(self.GAMMA) == False:
			raise ValueError("Gamma matrices not defined properly")
		#Construct B matrix and its inverse
		self.B = self.construct_b_matrix(PSI = self.PSI, GAMMA = self.GAMMA)
		self.B_inv = np.linalg.inv(self.B)

		#Construct M matrix
		self.M = np.einsum('ji,jkl->ikl',self.B_inv, self.GAMMA)

		#Trace of M
		self.TrM = np.einsum('...ii',self.M)

	def apply_waveplate(self, state, theta, phi):
		"""Apply a virtual waveplate to a state
		A general waveplate matrix is directly applied to a two-photon state.
		The state can be a pure state as well as a density matrix.

		:param state: The state onto which the waveplate is applied.
		:param theta: The angle of the fast axis with respect to self.H ([1,0])
		:param phi: The phase shift induced for light polarized along the slow axis

		:return: The resulting state or density matrix.
		"""
		w = [[np.cos(theta)**2+np.exp(1j*phi)*np.sin(theta)**2, 0.5*(np.exp(1j*phi)-1)*np.sin(2*theta)],
			[0.5*(np.exp(1j*phi)-1)*np.sin(2*theta), np.sin(theta)**2+np.exp(1j*phi)*np.cos(theta)**2]]
		w2 = np.kron(w, w)
		transformed_state = np.dot(w2, np.dot(state, np.conjugate(w2.T)))
		return transformed_state

	def rotate_reference_frame(self, state, theta):
		"""Apply a simple rotation to the state or density matrix.
		A special case would be rotating H to D and V to A by rotating 90°
		The state can be a pure state as well as a density matrix.

		:param state: The state onto which the rotation is applied.
		:param theta: The angle of rotation

		:return: The resulting state or density matrix.
		"""
		r = np.array(((np.cos(theta), -np.sin(theta)), (np.sin(theta), np.cos(theta))))
		r2 = np.kron(r, r)
		transformed = np.dot(r2, np.dot(state, np.conjugate(r2.T)))
		return transformed

	def state(self,string):
		"""Compute state from string.

		:param state: Two letter string.

		:return: State vector.

		Example:
			.. code-block:: python

				state = dm.state("HH")

			will be: :math:`state=\\begin{pmatrix}1 \\\ 0 \end{pmatrix}\otimes\\begin{pmatrix}1 \\\ 0 \end{pmatrix}`
		"""
		if len(string)!=2:
			raise ValueError("String not two characters long")

		basis_branch1 = self.basis_str_to_object(string[0])
		basis_branch2 = self.basis_str_to_object(string[1])
		return np.hstack(np.outer(basis_branch1,basis_branch2))

	def rho_state(self, state):
		"""Compute the density matrix of a pure state.
		The state is described either by linear superpositions of *self.state()* or :math:`\psi_\\nu` tensor elements.

		:param state: The state expressed as a linear combination of state tensor elements.

		:return: The corresponding density matrix.

		Example:
			If the basis in __init__(basis) was chosen as:

			.. code-block:: python

				basis =	['HH', 'HV','VV','VH','RH','RV','DV','DH','DR','DD','RD','HD','VD','VL','HL','RL']

			The Bell state: :math:`\\frac{1}{\\sqrt{2}}(\\lvert HH \\rangle + i \\lvert VV \\rangle)`

			is described in python code with above basis as

			.. code-block:: python

				HH=self.state("HH")
				VV=self.state("VV")

				state=1/sqrt(2)*(HH+1j*VV)


			or as:

			.. code-block:: python

				state=1/sqrt(2)*(self.PSI[0]+1j*self.PSI[2])

		"""

		proj = np.einsum('ij,j->i',np.conj(self.PSI),state)
		projProb = np.conj(proj) * proj
		return self.rho(projProb)

	def rho_state_optimized(self, state):
		"""Compute the density matrix of a pure state based on the maximum likelihood approach.
		Aim: To test the maximum likelihood function.
		The state is described by linear superpositions of *self.state()* or :math:`\psi_\\nu` tensor elements.

		:param state: The state expressed as a linear combination of state tensor elements.

		:return: The density matrix computed by the maximum likelihood approach.
		:rtype: numpy array

		Example:
			If the basis in __init__(basis) was chosen as:

			.. code-block:: python

				basis =	['HH', 'HV','VV','VH','RH','RV','DV','DH','DR','DD','RD','HD','VD','VL','HL','RL']

			The Bell state: :math:`\\frac{1}{\\sqrt{2}}(\\lvert HH \\rangle + i \\lvert VV \\rangle)`

			is described in python code with above basis as

			.. code-block:: python

				HH=self.state("HH")
				VV=self.state("VV")

				state=1/sqrt(2)*(HH+1j*VV)

				dm=DensityMatrix(basis)
				dm.rho_state_optimized(state)

			or as:

			.. code-block:: python

				state = 1/sqrt(2)*(self.PSI[0]+1j*self.PSI[2])

				dm = DensityMatrix(basis)
				dm.rho_state_optimized(state)
		"""

		proj = np.einsum('ij,j->i', np.conj(self.PSI), state)
		projProb = np.conj(proj) * proj

		rho = self.rho(projProb)
		return self.rho_max_likelihood(projProb, self.basis)

	def rho(self, correlation_counts):
		"""Compute the density matrix from measured correlation counts.

		:param array correlation_counts: An array containing the correlation counts sorted according to the elements in self.basis.

		:return: The density matrix.
		:rtype: numpy array

		Example:
			.. code-block:: python

				correlation_counts = np.array([34749, 324, 35805, 444, 16324, 17521, 13441, 16901, 17932, 32028, 15132, 17238, 13171, 17170, 16722, 33586])
				basis = ["HH","HV","VV","VH","RH","RV","DV","DH","DR","DD","RD","HD","VD","VL","HL","RL"]

			Data from: D. F. V. James et al. Phys. Rev. A, 64, 052312 (2001).
		"""
		NormFactor = np.dot(self.TrM, correlation_counts)
		rho = 1/NormFactor* np.einsum('i,ijk->jk',correlation_counts, self.M)
		return rho

	def construct_b_matrix(self, PSI, GAMMA):
		"""Construct B matrix as in D. F. V. James et al. Phys. Rev. A, 64, 052312 (2001).

		:param array PSI: :math:`\psi_\\nu` vector with :math:`\\nu=1,...,16`, computed in __init__
		:param array GAMMA: :math:`\Gamma` matrices, computed in __init__

		:return: :math:`B_{\\nu,\mu} = \\langle \psi_\\nu \\rvert  \Gamma_\mu  \\lvert \psi_\\nu \\rangle`
		:rtype: numpy array
		"""

		B = np.complex_(np.zeros((16,16)))

		for i in range(16):
			for j in range(16):
				B[i,j] = np.dot(np.conj(PSI[i]) , np.dot(GAMMA[j], PSI[i]))
		return B

	def test_gamma(self, gamma):
		"""Test if :math:`\Gamma_i, {i=1,...,16}` matrices are properly defined.

		:param array GAMMA: Gamma matrices.

		Test for:

			:math:`\mathrm{Tr}(\Gamma_i\Gamma_j)= \delta_{i,j}`

		:return: True if equation fullfilled for all gamma matrices, False otherwise.
		:rtype: bool
		"""

		#initialize empty test matrix
		test_matrix = np.complex_(np.zeros((16,16)))

		for i in range(len(gamma)):
			for j in range(len(gamma)):
				test_matrix[i,j] = np.trace(np.dot(gamma[i], gamma[j]))

		diag_matrix = np.diag(np.ones(16))


		test_result = np.einsum('ij,ij',test_matrix - diag_matrix, test_matrix - diag_matrix)-16

		if np.abs(test_result) < 10**-6:
			return True
		else:
			False

	def basis_str_to_object(self, pol = "H"):
		"""Relate string of basis element to Stokes vector.

		:param str pol: String of the measurement basis. Valid elements H, V, R, L, D, A.

		:return: Stokes vector of the polarization specified by H, V, R, L, D, A.
		:rtype: numpy array

		Example:

			.. code-block:: python

				pol = \"A\"
		"""

		if pol == "H":
			return self.H
		elif pol == "V":
			return self.V
		elif pol == "R":
			return self.R
		elif pol == "L":
			return self.L
		elif pol == "D":
			return self.D
		elif pol == "A":
			return self.A

		raise ValueError("Value not of the form H,V,R,L,A,D")

	def entropy_neumann(self, rho):
		"""Compute the von Neumann entropy of the density matrix.

		:param numpy_array rho: Density matrix

		:return: The von Neumann entropy of the density matrix. :math:`S=-\sum_{j}^{}m_j \mathrm{ln}(m_j)`, where :math:`m_j` denotes the eigenvalues of rho.
		:rtype: complex
		"""

		eigValues, eigVecors = np.linalg.eig(rho)
		S = 0
		for eigValue in eigValues:
			if eigValue == 0:
				#Catches the problem 0*ln(0) which is 0: lim_{x->0}x ln(x)=0.
				S = S + 0.0
			else:
				S = S + eigValue*np.log(eigValue)
		return -S

	def concurrence(self, rho):
		"""Compute the concurrence of the density matrix.

		:param numpy_array rho: Density matrix

		:return: The concurrence, see https://en.wikipedia.org/wiki/Concurrence_(quantum_computing).
		:rtype: complex
		"""

		rhoTilde = np.dot( np.dot(np.kron(self.PAULI[1], self.PAULI[1]), rho.conj()), np.kron(self.PAULI[1], self.PAULI[1]))

		rhoRoot = scipy.linalg.fractional_matrix_power(rho, 1/2.0)

		R = scipy.linalg.fractional_matrix_power(np.dot(np.dot(rhoRoot,rhoTilde),rhoRoot),1/2.0)

		eigValues, eigVecors = np.linalg.eig(R)
		sortedEigValues = np.sort(eigValues)

		con = sortedEigValues[3]-sortedEigValues[2]-sortedEigValues[1]-sortedEigValues[0]
		return np.max([con,0])

	def purity(self, rho):
		"""Compute the purity of the density matrix.

		:param numpy_array rho: Density matrix

		:return: The density matrix's purity :math:`\mathrm{Tr}\\rho^2`
		:rtype: complex
		"""
		return np.trace(np.dot(rho,rho))

	def fidelity(self, m, n):
		"""Compute the fidelity between the density matrices m and n.

		:param numpy_array m: Density matrix
		:param numpy_array n: Density matrix

		:return: The fideltiy between m and n (:math:`\mathrm{Tr}(\sqrt{\sqrt{m}n\sqrt{m}})^2`).
		:rtype: complex
		"""
		sqrt_m = fractional_matrix_power(m, 0.5)
		fidelity = np.trace(fractional_matrix_power(np.dot(sqrt_m, np.dot(n, sqrt_m)), 0.5))**2.0

		return fidelity

	def fidelity_purestate(self, rho, state):
		"""Compute the fidelity to a pure state. Should be faster than fidelity().

		:param numpy_array rho: Density matrix
		:param numpy_array state: state

		:return: The fidelity between self and state (:math:`\abs\mathrm{state}\\rho\mathrm{state}`).
		:rtype: float
		"""
		fidelity = np.abs(np.dot(np.dot(rho, state), np.conjugate(state)))

		return fidelity

	def fidelity_max(self, rho):
		"""Compute the maximal fidelity of rho to a maximally entangled state.

		:param numpy_array rho: Density matrix

		:return: The maximal fidelity of rho :math:`(\\rho)` to a maximally entangled state.
			:math:`F(\\rho)=\\frac{1+\lambda_1+\lambda_2-\mathrm{sgn}(\mathrm{det(R)})\lambda_3}{4}`, where

			:math:`R_{i,j}=\mathrm{Tr}(\sigma_i \otimes \sigma_j)`, with
			:math:`\sigma_i, {i=1,2,3}` the Pauli matrices and
			:math:`\lambda_i, {i=1,2,3}` the ordered singular values of R.

			Note, the maximally entangled state is not computed.
			Algorithm from: http://dx.doi.org/10.1103/PhysRevA.66.022307
		:rtype: complex
		"""

		R = np.complex_(np.zeros((3,3)))

		for i in range(3):
			for j in range(3):
				element = np.dot(rho , np.kron(self.PAULI[i] , self.PAULI[j]))
				#Trace of element
				R[i,j]  = np.trace(element)

		a, b = np.linalg.eig(R)

		SgnDetR = np.sign(np.linalg.det(R))

		#Singular Value of matrix are the square roots of the eigenvalus of A^* A, where A^* is the conjugate transpose of A.
		eigValues, eigVecors = np.linalg.eig(np.dot(R.conj().T, R))
		lmbda = np.sort(np.sqrt(eigValues))

		#The fidelity is:
		fidelity = (1+lmbda[2] + lmbda[1] - SgnDetR*lmbda[0])/4.0

		return fidelity

	def waveplate_rho_to_target(self, rho, target, theta_min=-np.pi/2, theta_max=np.pi/2, phi_min=-np.pi/2, phi_max=np.pi/2):
		"""Rotates input density matrix to target density matrix using a virtual waveplate.

		:param numpy_array rho: Density matrix
		:param numpy_array target: Target density matrix or state
		:param float theta_min: Lower bound for theta, default -pi/2
		:param float theta_max: Upper bound for theta, default pi/2
		:param float phi_min: Lower bound for phi, default -pi/2
		:param float phi_max: Upper bound for phi, default pi/2

		:return: closest match to rho_target and angles of virtual waveplates theta and phi.
		:rtype: tuple(numpy array, float, float)

		"""
		if type(target) != np.ndarray:
			target = np.array(target)
		if target.shape == (4, 4):
			optfun = self.opt_waveplate
		elif target.shape == (4,):
			optfun = self.opt_waveplate_to_state
		else:
			raise TypeError('Please use a state vector or density matrix as input')

		x0_array = np.array([0, 0])
		c_estimates = minimize(optfun, x0=x0_array, args=(rho, target), method='L-BFGS-B',
								bounds=[(theta_min, theta_max), (phi_min, phi_max)])
		theta = c_estimates.x[0]
		phi = c_estimates.x[1]
		rho_rotated = self.apply_waveplate(rho, theta, phi)
		if not c_estimates.success:
			print('minimizer for waveplate angles failed')
		return rho_rotated, theta, phi

	def opt_waveplate_to_state(self, coeff_array, rho, state_target):
		"""Helper function for *self.rotate_rho_to_target*.

		:param numpy_array coeff_array: Coefficient array, to be optimized
		:param numpy_array rho: Initial density matrix
		:param numpy_array state: Target state

		:return: 1-fidelity, such that the minimizing function finds the maximum of the fidelity.
		:rtype: complex
		"""
		sigma = self.apply_waveplate(rho, coeff_array[0], coeff_array[1])
		fidelity = np.real(self.fidelity_purestate(sigma, state_target))
		return 1-fidelity


	def opt_waveplate(self, coeff_array, rho, rho_target):
		"""Helper function for *self.rotate_rho_to_target*.

		:param numpy_array coeff_array: Coefficient array, to be optimized
		:param numpy_array rho: Initial density matrix
		:param numpy_array rho_target: Target density matrix

		:return: 1-fidelity, such that the minimizing function finds the maximum of the fidelity.
		:rtype: complex
		"""
		sigma = self.apply_waveplate(rho, coeff_array[0],coeff_array[1])
		fidelity = np.real(self.fidelity(sigma, rho_target))
		return 1-fidelity

	def find_closest_pure_state(self, rho, basis =["HH", "HV", "VH", "VV"]):
		"""Finds the closest pure state to the density matrix rho in the given basis.

		:param numpy_array rho: density matrix
		:param array basis: The basis in which the state is described

		:return: state vector describing the closest pure state. By convention the first vector element has vanishing complex component.
		:rtype: numpy array

		"""
		SPAN_RHO =[]

		for basis_element in basis:
			basis_branch1 = self.basis_str_to_object(basis_element[0])
			basis_branch2 = self.basis_str_to_object(basis_element[1])
			SPAN_RHO.append(np.hstack(np.outer(basis_branch1, basis_branch2)))

		x0_array =np.ones(8)

		c_estimates = minimize(self.opt_pure_state, x0 = x0_array, args = (rho, SPAN_RHO),method='L-BFGS-B')

		#state = np.dot(c_estimates, SPAN_RHO)
		coeff = []
		for i in range(4):
			coeff.append(c_estimates.x[2*i]+1j*c_estimates.x[2*i+1])

		state_norm = coeff / np.sqrt(np.vdot(coeff, coeff))

		#convention: first basis elemnt has no imaginary part
		theta =np.angle(state_norm[0])
		state_norm_conv = np.exp(-1j*theta)*state_norm
		if not c_estimates.success:
			print('minimizer for closest pure state failed')
		return state_norm_conv

	def opt_pure_state(self, coeff_array, rho, basis):
		"""Helper function for *self.find_closest_pure_state*.

		:param numpy_array coeff_array: Coefficient array, to be optimized
		:param numpy_array rho: Density matrix
		:param basis: The basis state from which the pure state is constructed, e.g. ["HH", "HV", "VH", "VV"]

		:return: 1-fidelity, such that the minimizing function finds the maximum of the fidelity.
		:rtype: complex
		"""
		coeff = []
		for i in range(4):
			coeff.append(coeff_array[2*i]+1j*coeff_array[2*i+1])

		coeff = coeff / np.sqrt(np.vdot(coeff, coeff))

		state =np.dot(coeff, basis)

		fidelity = np.real(self.fidelity_purestate(rho, state))

		return 1-fidelity

	def rho_max_likelihood(self, corr_counts, basis):
		"""Compute the density matrix based on the maximum likelihood approach.
		The minimum lenght of corr_counts and basis is 16. However, the quality of the estimation can be improved by performing the experiment in more bases. Maximum length is 36.

		:param numpy_array corr_counts:	Measured correlation counts corresponding to the basis.
		:param numpy_array basis: Basis in which correlations were measured.

		:return: Density matrix which is positive semidefinite.
		:rtype: numpy array
		"""

		#To get a start value for the optimization we are not using the Cholesky-decomposition since it fails for example for states like HH.
		#The reason is that in this case the Cholesky-decomposition of the density matrix fails because of its implementation.
		#We start with a flat distribution of all ones to make no assumptions about the states.
		t_array = np.ones(16)

		PSI=[]
		#Fill PSI
		for basis_element in basis:
			basis_branch1 = self.basis_str_to_object(basis_element[0])
			basis_branch2 = self.basis_str_to_object(basis_element[1])
			PSI.append(np.hstack(np.outer(basis_branch1,basis_branch2)))

		t_estimates = minimize(self.fun, x0=t_array, args=[corr_counts, basis, PSI], method='L-BFGS-B')
		if not t_estimates.success:
			print('minimizer for density matrix failed')
		return self.rho_phys(t_estimates.x)

	def fun(self, t, args):
		"""Maximum likelihood function to be minimized.

		:param numpy_array t: t values.
		:param numpy_array args: first entry contains correlation counts, second the corresponding basis as string.
		:param numpy_array args: PSIs, all possible basis as Jones vectors.

		:return: Function value. See for further information D. F. V. James et al. Phys. Rev. A, 64, 052312 (2001).
		:rtype: numpy float
		"""
		corr_counts=args[0]
		basis=args[1]
		PSI=args[2]
		nbrOfElements=len(basis)

		rho_phys = self.rho_phys(t)

		#Estimate NormFactor
		estNormFactor=[]

		for i in range(len(corr_counts)):
			estNormFactor.append(np.dot(np.dot(np.conj(PSI[i]), rho_phys),PSI[i]))

		NormFactor=np.sum(corr_counts)/np.sum(estNormFactor)

		#Optimize density matrix
		BraRoh_physKet = np.complex_(np.zeros(nbrOfElements))

		for i in range(nbrOfElements):
			rhoket = np.array(np.dot(rho_phys,PSI[i]).flat) #Convert 2d to 1d array with flat
			BraRoh_physKet[i] = np.complex_(np.dot(np.conj(PSI[i]), rhoket))

		return np.real(np.sum((NormFactor*BraRoh_physKet-corr_counts)**2.0/(2.0*NormFactor*BraRoh_physKet)))

	def rho_phys(self, t):
		"""Positive semidefinite matrix based on t values.

		:param numpy_array t: tvalues

		:return: A positive semidefinite matrix which is an estimation of the actual density matrix.
		:rtype: numpy matrix
		"""
		T = np.complex_(np.matrix([
						[t[0], 				0, 				0, 				0],
						[t[4]+1j*t[5], 		t[1], 			0, 				0],
						[t[10]+1j*t[11], 	t[6]+1j*t[7], 	t[2], 			0],
						[t[14]+1j*t[15], 	t[12]+1j*t[13], t[8]+1j*t[9], 	t[3]]
						]))

		TdagT = np.dot(T.conj().T , T)
		norm = np.trace(TdagT)

		return TdagT/norm

class Errorize(DensityMatrix):
	"""Compute +- uncertainty of the density matrix.
	A Monte Carlo simulation is performed based on counting statistics.

	:param array basis: Basis of measurements
	:param array cnts: Correlation counts of the measurements


	"""

	def __init__(self, basis, cnts):
		"""Initialize Errorize class. Can handle over defined > 16 basis measurements.

		:param array basis: Basis of measurements, length min 16 and max 36
		:param array cnts: Correlation counts of the measurements, length min 16 and max 36

		Example:

			.. code-block:: python

				cnts	= np.array([34749, 324, 35805, 444, 16324, 17521, 13441, 16901, 17932, 32028, 15132, 17238, 13171, 17170, 16722, 33586])
				basis   = ["HH","HV","VV","VH","RH","RV","DV","DH","DR","DD","RD","HD","VD","VL","HL","RL"]

			Data from: D. F. V. James et al. Phys. Rev. A, 64, 052312 (2001).
		"""
		self.cnts = cnts
		self.basis = basis

		self.b0 = ["HH", "HV", "VV", "VH", "RH", "RV", "DV", "DH", "DR", "DD", "RD", "HD", "VD", "VL", "HL", "RL"]

	def sim_counts(self, counts):
		"""Simulates counting statistics noise.

		:param numpy_array counts: Measured counts.

		:return: Array of simulated counting statistics values.
		:rtype: numpy array
		"""
		simc = np.random.normal(counts, np.sqrt(counts))
		return simc


	def __collect_results(self, result):
		"""Helper function for multicore processing.
		"""
		#rho, rhorec = result['rhos'], result['rhosrec']
		rhorec = result['rhosrec']
		#for r in rho:
		#	self.rhos.append(r)
		for r in rhorec:
			self.rhosrec.append(r)

	def sim(self, n_cycles_per_core, basis):
		"""Perform Monte Carlo simulation on one CPU.

		:param float n_cycles_per_core: Number of simulations per core.

		:return:
			a dictionary
				{'rhos': self.rhos, 'rhosrec': self.rhosrec} where
			self.rhos
				Array with raw density matrices.
			self.rhosrec
				Array with maximum likelihood approximated matrices.
		:rtype: dict
		"""
		possibleMatrices = []
		rhosrec = []

		for i in range(n_cycles_per_core):
			possibleCnts = self.sim_counts(self.cnts)
			possibleMatrices.append(DensityMatrix(self.b0))
			possibleMatrices[i].cnts = possibleCnts

		for m in possibleMatrices:
			rhosrec.append(m.rho_max_likelihood(m.cnts, basis))

		return {'rhosrec':rhosrec}

	def multiprocessing_simulate(self, n_cycles_per_core  = 10, nbr_of_cores = 8):
		"""Perform Monte Carlo simulation parallel on several CPU cores. Each core will call function self.sim().

		:param float n_cycles_per_core:  Number of simulations per core.
		:param float nbr_of_cores: Number of CPUs
		:return:
			self.rhos, self.rhosrec
				self.rhos
					array with raw density matrices and
				self.rhosrec
					array with maximum likelihood approximated matrices.

				Note: 'rhosrec' stands for rho reconstructed.

		:rtype: numpy matrices
		"""

		#self.rhos = []
		self.rhosrec = []

		pool = Pool()

		for i in range(nbr_of_cores):
			pool.apply_async(self.sim, [n_cycles_per_core, self.basis], callback = self.__collect_results)

		pool.close()
		pool.join()

		#return self.rhos, self.rhosrec
		return self.rhosrec

	def complex_std_dev(self, matrices):
		"""Compute the standard deviation for the real and complex part of matrices separately.

		:param numpy_array matrices: An array filled with matrices.

		:return: Standard deviation of the real and complex part for every matrix element.
		:rtype: complex
		"""
		return np.std(np.real(matrices), axis=0) + 1j*np.std(np.imag(matrices), axis=0)

	def fidelity_max(self):
		"""Compute the standard deviation of the maximal fidelity of :math:`\\rho` to a maximally entangled state.

		:returns: Its standard deviation.

		.. rubric:: Note, the maximally entangled state is not computed.
			Function from: http://dx.doi.org/10.1103/PhysRevA.66.022307

		"""
		d = DensityMatrix(self.b0)
		fidelities = []

		for r in self.rhosrec:
			fidelities.append(d.fidelity_max(r))

		std_dev = self.complex_std_dev(fidelities)
		return std_dev

	def concurrence(self):
		"""Compute the standard deviation of the concurrence of density matrix.

		:return: Its standard deviation.

		"""
		d = DensityMatrix(self.b0)
		c = []

		for r in self.rhosrec:
			c.append(d.concurrence(r))

		std_dev = self.complex_std_dev(c)
		return std_dev

	def purity(self):
		"""Compute the standard deviation of the density matrix's purity.

		:return: Its standard deviation.

		"""
		d = DensityMatrix(self.b0)
		c = []

		for r in self.rhosrec:
			c.append(d.purity(r))

		std_dev = self.complex_std_dev(c)
		return std_dev

	def rho_max_likelihood(self):
		"""Compute the standard deviation of the density matrix reconstructed by the maximum likelihood method.

		:return: Its standard deviation.

		"""
		std_dev = self.complex_std_dev(self.rhosrec)
		return std_dev

	def entropy_neumann(self):
		"""Compute the standard deviation of the von Neumann entropy.

		:param numpy_array rho: Density matrix

		:return: Its standard deviation.
		"""

		d = DensityMatrix(self.b0)
		c = []

		for r in self.rhosrec:
			c.append(d.entropy_neumann(r))

		std_dev = self.complex_std_dev(c)
		return std_dev

#If it is not used as a library, show how it works:
if __name__ == "__main__":
	import numpy as np

	round_digits =2

	#Compute the raw density matrix, data from: D. F. V. James et al. Phys. Rev. A, 64, 052312 (2001).
	basis = ["HH", "HV", "VV", "VH", "RH", "RV", "DV", "DH", "DR", "DD", "RD", "HD", "VD", "VL", "HL", "RL"]
	cnts = np.array([34749, 324, 35805, 444, 16324, 17521, 13441, 16901, 17932, 32028, 15132, 17238, 13171, 17170, 16722, 33586])

	dm = DensityMatrix(basis=basis)

	rho = dm.rho(cnts)
	#However, due to measurement imperfections this matrix is not necessarily positive semidefinite.
	#Find density matrix which best fits the data.

	rho_recon 	= dm.rho_max_likelihood(cnts, basis)


	closest_state_basis = ["HH", "HV", "VH", "VV"]
	closest_state = dm.find_closest_pure_state(rho_recon, basis=closest_state_basis)
	rho_state_closest = dm.rho_state(closest_state)

	s = str()
	for i in range(3):
		s = s + "\t"+ str(closest_state[i]) + "\t|"+closest_state_basis[i] + "> + \n"
	s = s + "\t" + str(closest_state[3]) + "\t|" + closest_state_basis[3] + ">"

	print("Basis: \n" + str(basis) + "\n")
	print("Cnts: \n" + str(cnts) + "\n")

	print("Raw Rho: \n" 						+ str(np.around(rho, decimals=round_digits)) + "\n")
	print("Rho Reconstructed: \n" 				+ str(np.around(rho_recon, decimals=round_digits)) + "\n")

	print("Closest State: \n" + s + "\n")

	print("Rho State closest: \n" 				+ str(np.around(rho_state_closest, decimals=round_digits)) + "\n")

	#Define states HH and VV
	HH = dm.state("HH")
	RR = dm.state("RR")
	RL = dm.state("RL")
	LR = dm.state("LR")
	LL = dm.state("LL")
	VV = dm.state("VV")

	print('A virtual waveplate can be applied to transform the result to the Bell state HH - VV\n')
	# calculate waveplate to transform HH+VV to HH-VV
	rho_auto_tranformed, theta, phi = dm.waveplate_rho_to_target(rho_recon, dm.rho_state(state=1 / np.sqrt(2) * (HH - VV)))
	# if angles are known they can also be applied directly
	rho_manually_transformed = dm.apply_waveplate(rho_recon, theta, phi)
	print("Rho Auto Transformed: \n" 						+ str(np.around(rho_auto_tranformed, decimals=round_digits)) + "\n")
	print(f'theta: {np.around(theta/np.pi, decimals=round_digits+1)} Pi')
	print(f'phi: {np.around(phi/np.pi, decimals=round_digits+1)} Pi\n')
	print("Rho Manually Transformed: \n" 						+ str(np.around(rho_manually_transformed, decimals=round_digits)) + "\n")

	print("Rho of Bell state HH + iVV: \n" 		+ str(np.around(dm.rho_state(state=1/np.sqrt(2)*(HH+1j*VV)), decimals=round_digits))+"\n")

	print("Entropy of HH + iVV:" 				+ str(np.around(dm.entropy_neumann(dm.rho_state(state=1/np.sqrt(2)*(HH+1j*VV))), decimals=round_digits))+"\n")

	print("Concurrence of HH + iVV:" 			+ str(np.around(dm.concurrence(dm.rho_state(state=1/np.sqrt(2.0)*(HH+1.0j*VV))), decimals=round_digits))+"\n")
	print("Concurrence of HH: " 				+ str(np.around(dm.concurrence(dm.rho_state(state=HH)), decimals=round_digits)) +"\n")

	print("Density matrix of RR + RL + LR +LL: \n" 			+ str(np.around(dm.rho_state(state=1/4.0*(RR + LR + RL + LL)), decimals=round_digits)) +"\n")

	print("Optimized Density matrix of HH: \n" 	+ str(np.around(dm.rho_state_optimized(state=HH), decimals=round_digits)) +"\n")

	print("Fidelity of HH + iVV:" 	    		+ str(np.around(dm.fidelity_max(dm.rho_state(state=1/np.sqrt(2)*(HH+1j*VV))), decimals=round_digits))+"\n")

	print("Fidelity of HH: " 	    			+ str(np.around(dm.fidelity_max(dm.rho_state(state=HH)), decimals=round_digits))+"\n")

	eigValues, eigVecors = np.linalg.eig(rho_recon)

	print("Eigen-values: " + str(np.around(eigValues, decimals=round_digits))+"\n")
	print("Pureness: " + str(np.around(np.trace(np.dot(rho_recon, rho_recon)), decimals=round_digits)) +"\n")
	print("von Neumann Entropy: \n" + str(np.around(dm.entropy_neumann(rho_recon), decimals=round_digits))+"\n")

	fidelity = dm.fidelity_max(rho_recon)
	print("Fidelity: " + str(fidelity)+"\n")

	r = dm.rho_max_likelihood(np.array([34749, 324, 35805, 444, 16324, 17521, 13441, 16901, 17932, 32028, 15132, 17238, 13171, 17170, 16722, 33586]),
								["HH", "HV", "VV", "VH", "RH", "RV", "DV", "DH", "DR", "DD", "RD", "HD", "VD", "VL", "HL", "RL"])
	print(dm.fidelity_max(r))
	eigValues, eigVecors = np.linalg.eig(r)
	print("Eigen-values: " + str(np.around(eigValues, decimals=round_digits))+"\n")

	#Calculate Error
	fidelity    	= dm.fidelity_max(rho_recon)
	concurrence 	= dm.concurrence(rho_recon)
	purity			= dm.purity(rho_recon)
	entropyNeumann  = dm.entropy_neumann(rho_recon)

	#Compute errors
	err = Errorize(basis, cnts)
	#t0=time.time()
	err.multiprocessing_simulate(n_cycles_per_core=10, nbr_of_cores=2)

	rho_err = err.rho_max_likelihood()

	print("Rho Max Likelhood: \n"               + str(np.around(rho_recon, decimals=round_digits)) + "\n")
	print("and its error: \n"                   + str(np.around(rho_err, decimals=round_digits)) + "\n")

	fid_std = np.around(err.fidelity_max(), decimals=5)
	con_std = np.around(err.concurrence(), decimals=5)
	pur_std = np.around(err.purity(), decimals=5)
	ent_std = np.around(err.entropy_neumann(), decimals=5)

	#print(str(time.time()-t0))
	print("Fidelity: " +    str(fidelity)    + " +- " + str(fid_std))
	print("Concurrence: " + str(concurrence) + " +- " + str(con_std))
	print("Purity: " + str(purity) + " +- " + str(pur_std))
	print("Von Neumann entropy: " + str(entropyNeumann) + " +- " + str(ent_std))
