"""
Author: Adrian Sosic
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from opt_einsum import contract
from abc import ABC
from itertools import zip_longest
from utils import normalize, shift
from typing import Optional
plt.style.use('seaborn')


class TransformInvariantNMF(ABC):
	"""Abstract base class for transform-invariant non-negative matrix factorization."""

	def __init__(
			self,
			atom_size: Optional[int],
			n_components: int = 10,
			sparsity_H: float = 0.1,
			refit_H: bool = True,
			n_iterations: int = 100,
			eps: float = 1e-9
	):
		"""
		Parameters
		----------
		atom_size : int
			Dimensionality of a single dictionary element.
		n_components : int
			Dictionary size (= number of dictionary elements).
		sparsity_H : float
			Regularization parameter for the activation tensor.
		refit_H : bool
			If True, the activation tensor gets refitted using the learned dictionary to mitigate amplitude bias.
		n_iterations : int
			Number of learning iterations.
		eps : float
			Small constant to avoid division by zero.
		"""
		# store parameters
		self.atom_size = atom_size
		self.n_components = n_components
		self.n_iterations = n_iterations
		self.sparsity = sparsity_H
		self.refit_H = refit_H

		# signal, reconstruction, factorization, and transformation matrices
		self.V = None
		self._R = None
		self._T = None
		self._W = None
		self._H = None

		# constant to avoid division by zero
		self.eps = eps

		# caching flags
		self._is_ready_R = False

	@property
	def R(self) -> np.array:
		"""The reconstructed signal matrix."""
		if not self._is_ready_R:
			self._R = self._reconstruct()
			self._is_ready_R = True
		return self._R

	@property
	def W(self) -> np.array:
		"""The dictionary matrix."""
		return self._W

	@W.setter
	def W(self, W: np.array):
		self._W = W
		self._is_ready_R = False

	@property
	def H(self) -> np.array:
		"""The activation tensor."""
		return self._H

	@H.setter
	def H(self, H: np.array):
		self._H = H
		self._is_ready_R = False

	@property
	def n_dim(self) -> int:
		"""Number of input dimensions."""
		return self.V.shape[0]

	@property
	def n_signals(self) -> int:
		"""Number of input signals."""
		return self.V.shape[1]

	@property
	def n_transforms(self) -> int:
		"""Number of dictionary transforms."""
		return len(self._T)

	def _reconstruct(self) -> np.array:
		"""Reconstructs the signal matrix generically using transformation matrices."""
		return contract('tdh,hm,tmn->dn', self._T, self.W, self.H, optimize='optimal')

	def generate_transforms(self) -> np.array:
		"""Generates all dictionary transforms for the given signal matrix."""
		raise NotImplementedError

	def initialize(self, V):
		"""
		Stores the signal matrix and initialize the factorization (and transformation) matrices.

		Notation:
		---------
		d: number of input dimensions
		n: number of input samples
		m: number of basis vectors (dictionary size)
		t: number of basis vector transforms (= 1 for standard NMF without transform invariance)
		h: number of basis vector dimensions (= d for standard NMF without transform invariance)

		Dimensions:
		-----------
		Signal matrix V: 		d x n
		Dictionary Matrix W: 	h x m
		Activation Tensor H: 	t x m x n
		Transformation Tensor:  t x d x h
		"""
		# store the signal matrix
		self.V = np.asarray(V)

		# if explicit transformation matrices are used, create and store them
		try:
			self._T = self.generate_transforms()
		except NotImplementedError:
			pass

		# initialize the factorization matrices
		self.W = normalize(np.random.random([self.atom_size, self.n_components]))
		self.H = np.random.random([self.n_transforms, self.n_components, self.n_signals])

	def fit(self, V):
		"""Learns an NMF representation of a given signal matrix."""
		# initialize all matrices
		self.initialize(V)

		# TODO: define stopping criterion
		# iterate the multiplicative update rules
		for i in range(self.n_iterations):
			print(f"Iteration: {i}\tReconstruction error: {norm(self.V - self.R, 'fro')}")
			self.update_H()
			self.update_W()

		# TODO: define stopping criterion
		# refit the activations using the learned dictionary
		if self.refit_H:
			for i in range(10):
				self.update_H(sparsity=False)

	def _reconstruction_gradient_H(self) -> np.array:
		"""Positive and negative parts of the gradient of the reconstruction error w.r.t. the activation tensor."""
		TW = contract('tdh,hm->tdm', self._T, self.W, optimize='optimal')
		numer = contract('tdm,dn->tmn', TW, self.V, optimize='optimal')
		denum = contract('tdm,dn->tmn', TW, self.R, optimize='optimal')
		return numer, denum

	def update_H(self, sparsity: bool = True):
		"""
		Multiplicative update of the activation tensor.

		Parameters
		----------
		sparsity : bool
			If True, sparsity regularization is applied.
		"""
		# compute the gradients of the reconstruction error
		numer, denum = self._reconstruction_gradient_H()

		# add sparsity regularization
		if sparsity:
			denum = denum + self.sparsity

		# update the activation tensor
		self.H = self.H * (numer / (denum + self.eps))

	def _reconstruction_gradient_W(self) -> np.array:
		"""Positive and negative parts of the gradient of the reconstruction error w.r.t. the dictionary matrix."""
		numer = contract('tdh,dn,tmn->hm', self._T, self.V, self.H, optimize='optimal')
		denum = contract('tdh,dn,tmn->hm', self._T, self.R, self.H, optimize='optimal')
		return numer, denum

	def update_W(self):
		"""Multiplicative update of the dictionary matrix."""
		# compute the gradients of the reconstruction error
		numer, denum = self._reconstruction_gradient_W()

		# update the dictionary matrix
		self.W = normalize(self.W * (numer / (denum + self.eps)))

	def plot_dictionary(self):
		"""Plots the learned dictionary elements."""
		fig, axs = plt.subplots(nrows=int(np.ceil(self.W.shape[1] / 3)), ncols=3)
		for w, ax in zip_longest(self.W.T, axs.ravel()):
			if w is not None:
				ax.plot(w)
			ax.axis('off')
		plt.tight_layout()
		return fig


class SparseNMF(TransformInvariantNMF):
	"""Class for sparse non-negative matrix factorization (special case of a transform invariant NMF with a single
	identity transformation and an atom size that equals the signal dimension)."""

	def __init__(self, **kwargs):
		super().__init__(atom_size=None, **kwargs)

	def initialize(self, X):
		"""Creates a TransformInvariantNMF where the atom size equals the signal size."""
		self.atom_size = np.shape(X)[0]
		super().initialize(X)

	def generate_transforms(self):
		"""No transformations are applied (achieved via a single identity transform)."""
		return np.eye(self.n_dim)[None, :, :]


class ExplicitShiftInvariantNMF(TransformInvariantNMF):
	"""Class for shift-invariant non-negative matrix factorization of 1-D signals that computes the involved
	transform operations explicitly via transformation matrices."""

	def generate_transforms(self) -> np.array:
		"""Generates all possible shift matrices for the signal dimension and given atom size."""
		# assert that the dictionary elements are at most as large as the signal
		assert self.atom_size <= self.n_dim

		# create the transformation that places the dictionary element at the beginning, and then shift it
		base_array = np.hstack([np.eye(self.atom_size), np.zeros([self.atom_size, self.n_dim - self.atom_size])])
		T = np.array([shift(base_array, s).T for s in range(-self.atom_size + 1, self.n_dim)])
		return T


class ImplicitShiftInvariantNMF(TransformInvariantNMF):
	"""Class for shift-invariant non-negative matrix factorization of 1-D signals that computes the involved
	transform operations implicitly via correlation/convolution."""

	@property
	def n_transforms(self) -> int:
		"""Number of dictionary transforms."""
		# TODO: inherit docstring from superclass
		return self.n_dim + self.atom_size - 1

	def _reconstruct(self) -> np.array:
		"""Reconstructs the signal matrix via convolution."""
		# TODO: vectorize
		# TODO: computation via FFT
		R = np.zeros([self.n_dim, self.n_signals])
		for n in range(self.n_signals):
			R[:, n] = np.sum([np.convolve(self.H[:, m, n], self.W[:, m], mode='valid')
							  for m in range(self.n_components)], axis=0)
		return R

	def _reconstruction_gradient_H(self) -> np.array:
		"""Positive and negative parts of the gradient of the reconstruction error w.r.t. the activation tensor."""
		# TODO: inherit docstring from superclass
		# TODO: vectorize
		# TODO: computation via FFT
		numer = np.zeros([self.n_transforms, self.n_components, self.n_signals])
		denum = np.zeros([self.n_transforms, self.n_components, self.n_signals])
		for n in range(self.n_signals):
			for m in range(self.n_components):
				numer[:, m, n] = np.correlate(self.V[:, n], self.W[:, m], mode='full')
				denum[:, m, n] = np.correlate(self.R[:, n], self.W[:, m], mode='full')
		return numer, denum

	def _reconstruction_gradient_W(self) -> np.array:
		"""Positive and negative parts of the gradient of the reconstruction error w.r.t. the dictionary matrix."""
		# TODO: inherit docstring from superclass
		# TODO: vectorize
		# TODO: computation via FFT
		numer = np.zeros([self.atom_size, self.n_components])
		denum = np.zeros([self.atom_size, self.n_components])
		for m in range(self.n_components):
			numer[:, m] = np.sum([np.correlate(self.V[:, n], self.H[:, m, n], mode='valid')
								  for n in range(self.n_signals)], axis=0)
			denum[:, m] = np.sum([np.correlate(self.R[:, n], self.H[:, m, n], mode='valid')
								  for n in range(self.n_signals)], axis=0)
		return numer, denum


class ShiftInvariantNMF(ExplicitShiftInvariantNMF, ImplicitShiftInvariantNMF):
	"""Wrapper class for shift-invariant non-negative matrix factorization of 1-D signals."""

	def __new__(cls, explicit_transforms: bool = False, **kwargs):
		if explicit_transforms:
			return ExplicitShiftInvariantNMF(**kwargs)
		else:
			return ImplicitShiftInvariantNMF(**kwargs)
