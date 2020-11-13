"""
Author: Adrian Sosic
"""

import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
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
			Regularization parameter for the activation matrix.
		refit_H : bool
			If True, the activation matrix gets refitted using the learned dictionary to mitigate amplitude bias.
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

		# signal, factorization, and transformation matrices
		self.V = None
		self.T = None
		self.W = None
		self.H = None

		# constant to avoid division by zero
		self.eps = eps

	@property
	def R(self) -> np.array:
		"""The reconstructed signal matrix."""
		# TODO: vectorize
		# TODO: avoid recomputation by using getter method
		R = np.zeros_like(self.V)
		for t in range(self.n_transforms):
			R = R + self.T[t] @ self.W @ self.H[t]
		return R

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
		return len(self.T)

	@abstractmethod
	def generate_transforms(self) -> np.array:
		"""Generates all dictionary transforms for the given signal matrix."""
		pass

	def initialize(self, V):
		"""Stores the signal matrix and initialize the factorization and transformation matrices."""
		self.V = np.asarray(V)
		self.T = self.generate_transforms()
		self.W = normalize(np.random.random([self.atom_size, self.n_components]))
		self.H = np.random.random([self.n_transforms, self.n_components, self.n_signals])

	def fit(self, V):
		"""Learns an NMF representation of a given signal matrix."""
		# initialize all matrices
		self.initialize(V)

		# TODO: define stopping criterion
		# iterate the multiplicative update rules
		for i in range(self.n_iterations):
			self.update_H()
			self.update_W()

		# TODO: define stopping criterion
		# refit the activations using the learned dictionary
		if self.refit_H:
			for i in range(10):
				self.update_H(sparsity=False)

	def update_H(self, sparsity: bool = True):
		"""
		Multiplicative update of the activation matrix.

		Parameters
		----------
		sparsity : bool
			If True, sparsity regularization is applied.
		"""
		# TODO: implement getter to avoid explicit precomputation
		# get the current reconstruction matrix
		R = self.R

		# TODO: vectorize
		# compute the gradients of the reconstruction error
		numer = np.zeros_like(self.H)
		denum = np.zeros_like(self.H)
		for t in range(self.n_transforms):
			TW = self.T[t] @ self.W
			numer[t] = TW.T @ self.V
			denum[t] = TW.T @ R

		# add sparsity regularization
		if sparsity:
			denum = denum + self.sparsity

		# update the activation matrix
		self.H = self.H * (numer / (denum + self.eps))

	def update_W(self):
		"""Multiplicative update of the dictionary matrix."""
		# TODO: implement getter to avoid explicit precomputation
		# get the current reconstruction matrix
		R = self.R

		# TODO: vectorize
		# compute the gradients of the reconstruction error
		numer = np.zeros_like(self.W)
		denum = np.zeros_like(self.W)
		for t in range(self.n_transforms):
			numer = numer + self.T[t].T @ self.V @ self.H[t].T
			denum = denum + self.T[t].T @ R @ self.H[t].T

		# update the dictionary matrix
		self.W = normalize(self.W * (numer / (denum + self.eps)))

	def plot_dictionary(self):
		"""Plots the learned dictionary elements."""
		fig, axs = plt.subplots(nrows=int(np.ceil(self.W.shape[1]/3)), ncols=3)
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


class ShiftInvariantNMF(TransformInvariantNMF):
	"""Class for shift-invariant non-negative matrix factorization of 1-D signals."""

	def generate_transforms(self) -> np.array:
		"""Generates all possible shift matrices for the signal dimension and given atom size."""
		# assert that the dictionary elements are at most as large as the signal
		assert self.atom_size <= self.n_dim

		# create the transformation that places the dictionary element at the beginning, and then shift it
		base_array = np.hstack([np.eye(self.atom_size), np.zeros([self.atom_size, self.n_dim - self.atom_size])])
		T = np.array([shift(base_array, s).T for s in range(-self.atom_size + 1, self.n_dim)])
		return T
