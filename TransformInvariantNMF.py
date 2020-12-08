"""
Author: Adrian Sosic
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import as_strided
from scipy.fft import fftn, ifftn, next_fast_len
from scipy.ndimage import convolve1d
from opt_einsum import contract
from itertools import product
from abc import ABC
from utils import normalize, shift
from typing import Optional, Tuple
plt.style.use('seaborn')

# TODO: replace 'matrix' with 'tensor' in docstrings
# TODO: indicate 'override' in subclasses
# TODO: refactor fft code parts into functions


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

		# axis over which the dictionary matrix gets normalized
		self._normalization_dims = 0

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
		return self.V.shape[-1]
	
	@property
	def n_channels(self) -> int:
		"""Number of input channels."""
		return self.V.shape[-2]

	@property
	def n_transforms(self) -> int:
		"""Number of dictionary transforms."""
		return len(self._T)

	def _reconstruct(self) -> np.array:
		"""Reconstructs the signal matrix generically using transformation matrices."""
		return contract('tdh,hcm,tmn->dcn', self._T, self.W, self.H, optimize='optimal')

	def generate_transforms(self) -> np.array:
		"""Generates all dictionary transforms for the given signal matrix."""
		raise NotImplementedError

	def initialize(self, V):
		"""
		Stores the signal matrix and initialize the factorization (and transformation) matrices.

		Notation:
		---------
		d: number of input dimensions
		c: number of input channels
		n: number of input samples
		m: number of basis vectors (dictionary size)
		t: number of basis vector transforms (= 1 for standard NMF without transform invariance)
		h: number of basis vector dimensions (= d for standard NMF without transform invariance)

		Dimensions:
		-----------
		Signal matrix V: 		d x c x n
		Dictionary Matrix W: 	h x c x m
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
		self._init_factorization_matrices()

	def _init_factorization_matrices(self):
		"""Initializes the activation matrix and dictionary matrix."""
		# TODO: use clever scaling of tensors for initialization
		self.W = normalize(np.random.random([self.atom_size, self.n_channels, self.n_components]), axis=self._normalization_dims)
		self.H = np.random.random([self.n_transforms, self.n_components, self.n_signals])

	def fit(self, V):
		"""Learns an NMF representation of a given signal matrix."""
		# initialize all matrices
		self.initialize(V)

		# TODO: define stopping criterion
		# iterate the multiplicative update rules
		for i in range(self.n_iterations):
			print(f"Iteration: {i}\tReconstruction error: {np.sqrt(np.sum((self.V - self.R) ** 2))}")
			self.update_H()
			self.update_W()

		# TODO: define stopping criterion
		# refit the activations using the learned dictionary
		if self.refit_H:
			for i in range(10):
				self.update_H(sparsity=False)

	def _reconstruction_gradient_H(self) -> (np.array, np.array):
		"""Positive and negative parts of the gradient of the reconstruction error w.r.t. the activation tensor."""
		TW = contract('tdh,hcm->tdcm', self._T, self.W, optimize='optimal')
		numer = contract('tdcm,dcn->tmn', TW, self.V, optimize='optimal')
		denum = contract('tdcm,dcn->tmn', TW, self.R, optimize='optimal')
		return numer, denum

	def _gradient_H(self, sparsity: bool = True) -> (np.array, np.array):
		"""
		Computes the positive and the negative parts of the energy gradient w.r.t. the activation tensor.

		Parameters
		----------
		sparsity : bool
			If True, the output includes the gradient of the sparsity regularization.

		Returns
		-------
		(numer, denum) : (np.array, np.array)
			The gradient components.
		"""
		# compute the gradients of the reconstruction error
		numer, denum = self._reconstruction_gradient_H()

		# add sparsity regularization
		if sparsity:
			denum = denum + self.sparsity

		return numer, denum

	def update_H(self, sparsity: bool = True):
		"""
		Multiplicative update of the activation tensor.

		Parameters
		----------
		sparsity : bool
			If True, sparsity regularization is applied.
		"""
		# compute the gradient components
		numer, denum = self._gradient_H(sparsity)

		# update the activation tensor
		self.H = self.H * (numer / (denum + self.eps))

	def _reconstruction_gradient_W(self) -> (np.array, np.array):
		"""Positive and negative parts of the gradient of the reconstruction error w.r.t. the dictionary matrix."""
		numer = contract('tdh,dcn,tmn->hcm', self._T, self.V, self.H, optimize='optimal')
		denum = contract('tdh,dcn,tmn->hcm', self._T, self.R, self.H, optimize='optimal')
		return numer, denum

	def update_W(self):
		"""Multiplicative update of the dictionary matrix."""
		# compute the gradients of the reconstruction error
		numer, denum = self._reconstruction_gradient_W()

		# update the dictionary matrix
		self.W = normalize(self.W * (numer / (denum + self.eps)), axis=self._normalization_dims)


class SparseNMF(TransformInvariantNMF):
	"""Class for sparse non-negative matrix factorization (special case of a transform invariant NMF with a single
	identity transformation and an atom size that equals the signal dimension)."""

	def __init__(self, **kwargs):
		super().__init__(atom_size=None, **kwargs)

	def initialize(self, X):
		"""Creates a TransformInvariantNMF where the atom size equals the signal size."""
		self.atom_size = np.shape(X)[0]
		super().initialize(X)

	def generate_transforms(self) -> np.array:
		"""No transformations are applied (achieved via a single identity transform)."""
		return np.eye(self.n_dim)[None, :, :]


class BaseShiftInvariantNMF(TransformInvariantNMF):
	"""Base class for shift-invariant non-negative matrix factorization."""

	def __init__(self, inhibition_range: Optional[int] = None, inhibition_strength: float = 0.1, **kwargs):
		"""
		Parameters
		----------
		inhibition_range : int
			Number of neighboring activation elements in each direction that exert an inhibitory effect.
			If 'None', the range is set to the minimal range that covers the size of a dictionary element.
		"""
		# set the basic parameters
		super().__init__(**kwargs)
		
		# default inhibition range = minimal range to cover the atom size
		if inhibition_range is None:
			inhibition_range = int(np.ceil(self.atom_size / 2))

		# store the inhibition parameters and construct the inhibition kernel
		self.inhibition_range = inhibition_range
		self.inhibition_strength = inhibition_strength
		self.kernel = 1 - ((np.arange(-inhibition_range, inhibition_range + 1) / inhibition_range) ** 2)

	@property
	def n_dim(self) -> Tuple[int]:
		"""Number of input dimensions."""
		return tuple(self.V.shape[:-2])

	@property
	def n_transforms(self) -> Tuple[int]:
		"""Number of dictionary transforms."""
		# TODO: inherit docstring from superclass
		return tuple(np.array(self.n_dim) + self.atom_size - 1)

	@property
	def n_shift_dimensions(self):
		"""The number of shift invariant input dimensions."""
		return self.V.ndim - 2

	@property
	def shift_dimensions(self):
		"""The dimension index of the shift invariant input dimensions."""
		return tuple(range(self.n_shift_dimensions))

	def initialize(self, V):
		# TODO: write getter for _normalization_dims
		super().initialize(V)
		self._normalization_dims = self.shift_dimensions
		self._init_cache()

	def _init_cache(self):
		"""Caches several fitting related variables."""
		cache = {}

		# if self._use_fft:
		if True:
			# TODO: verify that these are the optimal fft shapes
			cache['fft_shape_full'] = [next_fast_len(s) for s in np.array(self.V.shape[:self.n_shift_dimensions]) + np.array(self.W.shape[:self.n_shift_dimensions]) - 1]
			cache['fft_shape_valid'] = [next_fast_len(s) for s in np.array(self.V.shape[:self.n_shift_dimensions]) + np.array(self.H.shape[:self.n_shift_dimensions]) - 1]

		if True:
			# zero-padding of the signal matrix for full-size correlation
			cache['pad_width'] = (*self.n_shift_dimensions*((self.atom_size-1,)*2,), (0,0), (0,0))
			cache['V_padded'] = np.pad(self.V, pad_width=cache['pad_width'])

			# dimension labels of the data and reconstruction matrices
			cache['V_labels'] = ['d' + str(i) for i in range(self.n_shift_dimensions)] + ['c', 'n']
			cache['W_labels'] = ['a' + str(i) for i in range(self.n_shift_dimensions)] + ['c', 'm']
			cache['H_labels'] = ['d' + str(i) for i in range(self.n_shift_dimensions)] + ['m', 'n']

			# dimension info for striding in gradient_H computation
			cache['X_strided_W_shape'] = (self.atom_size,) * self.n_shift_dimensions + self.H.shape[:-2] + self.V.shape[-2:]
			cache['X_strided_W_strides'] = cache['V_padded'].strides[:self.n_shift_dimensions] + cache['V_padded'].strides
			cache['X_strided_W_labels'] = [s + str(i) for s, i in product(['a', 'd'], range(self.n_shift_dimensions))] + ['c', 'n']

			# dimension info for striding in gradient_W computation
			cache['H_strided_V_shape'] = self.V.shape[:self.n_shift_dimensions] + (self.atom_size,) * self.n_shift_dimensions + self.H.shape[-2:]
			cache['H_strided_V_strides'] = self.H.strides[:self.n_shift_dimensions] + self.H.strides
			cache['H_strided_V_labels'] = [s + str(i) for s, i in product(['d', 'a'], range(self.n_shift_dimensions))] + ['m', 'n']

			# dimension info for striding in reconstruction computation
			cache['H_strided_W_shape'] = (self.atom_size,) * self.n_shift_dimensions + self.V.shape[:-2] + self.H.shape[-2:]
			cache['H_strided_W_strides'] = self.H.strides[:self.n_shift_dimensions] + self.H.strides
			cache['H_strided_W_labels'] = [s + str(i) for s, i in product(['a', 'd'], range(self.n_shift_dimensions))] + ['m', 'n']

		self._cache = cache

	def _init_factorization_matrices(self):
		"""Initializes the activation matrix and dictionary matrix."""
		# TODO: inherit docstring from superclass
		self.W = normalize(np.random.random([*[self.atom_size] * self.n_shift_dimensions, self.n_channels, self.n_components]), axis=self._normalization_dims)
		self.H = np.random.random([*self.n_transforms, self.n_components, self.n_signals])

	def _gradient_H(self, sparsity: bool = True) -> (np.array, np.array):
		"""Computes the positive and the negative parts of the energy gradient w.r.t. the activation tensor."""
		# TODO: inherit docstring from superclass

		# compute the gradient w.r.t. the reconstruction and sparsity energies
		numer, denum = super()._gradient_H(sparsity)

		# add the inhibition gradient component
		if self.inhibition_range and self.inhibition_strength:
			inhibition = self.H.copy()
			for dim in range(self.n_shift_dimensions):
				inhibition = convolve1d(inhibition, self.kernel, axis=dim, mode='constant', cval=0.0)
			inhibition = inhibition - self.H
			denum = denum + self.inhibition_strength * inhibition

		return numer, denum


class ShiftInvariantNMF(BaseShiftInvariantNMF, ABC):
	"""Wrapper class for shift-invariant non-negative matrix factorization."""

	def __new__(cls, explicit_transforms: bool = False, **kwargs):
		if explicit_transforms:
			return ExplicitShiftInvariantNMF(**kwargs)
		else:
			return ImplicitShiftInvariantNMF(**kwargs)


class ExplicitShiftInvariantNMF(BaseShiftInvariantNMF):
	"""Class for shift-invariant non-negative matrix factorization that computes the involved transform operations
	explicitly via transformation matrices."""

	def initialize(self, V):
		if V.ndim > 3:
			raise ValueError("'ExplicitShiftInvariantNMF' currently supports (multi-channel) 1-D signals only. "
							 "For higher-dimensional signals, use 'ImplicitShiftInvariantNMF'.")
		super().__init__(V)

	def generate_transforms(self) -> np.array:
		"""Generates all possible shift matrices for the signal dimension and given atom size."""
		# assert that the dictionary elements are at most as large as the signal
		assert self.atom_size <= self.n_dim

		# create the transformation that places the dictionary element at the beginning, and then shift it
		base_array = np.hstack([np.eye(self.atom_size), np.zeros([self.atom_size, self.n_dim - self.atom_size])])
		T = np.array([shift(base_array, s).T for s in range(-self.atom_size + 1, self.n_dim)])
		return T


class ImplicitShiftInvariantNMF(BaseShiftInvariantNMF):
	"""Class for shift-invariant non-negative matrix factorization that computes the involved transform operations
	implicitly via correlation/convolution."""

	def __init__(self, use_fft=True, **kwargs):
		super().__init__(**kwargs)
		self._use_fft = use_fft

	def _reconstruct(self) -> np.array:
		"""Reconstructs the signal matrix via (fft-)convolution."""
		# TODO: switch original/frequency domains only on demand via setter/getters
		if self._use_fft:
			H_fft = fftn(self.H, axes=self.shift_dimensions, s=self._cache['fft_shape_full'])
			W_fft = fftn(self.W, axes=self.shift_dimensions, s=self._cache['fft_shape_full'])
			R_fft = contract('...cm,...mn->...cn', W_fft, H_fft)
			R_pad = np.real(ifftn(R_fft, axes=self.shift_dimensions))
			lower_idx = np.array(self.W.shape[:self.n_shift_dimensions]) - 1
			upper_idx = np.array(self.V.shape[:self.n_shift_dimensions]) + np.array(self.W.shape[:self.n_shift_dimensions]) - 1
			slices = tuple(slice(lower, upper) for lower, upper in zip(lower_idx, upper_idx))
			R = R_pad[slices]
		else:
			H_strided = as_strided(self.H, self._cache['H_strided_W_shape'], self._cache['H_strided_W_strides'], writeable=False)
			R = contract(H_strided, self._cache['H_strided_W_labels'], np.flip(self.W, range(self.n_shift_dimensions)), self._cache['W_labels'], self._cache['V_labels'], optimize='optimal')
		return R

	def _reconstruction_gradient_H(self) -> np.array:
		"""Positive and negative parts of the gradient of the reconstruction error w.r.t. the activation tensor."""
		# TODO: inherit docstring from superclass
		# TODO: switch original/frequency domains only on demand via setter/getters
		if self._use_fft:

			def fft_convolve(X):
				X_fft = fftn(X, axes=self.shift_dimensions, s=self._cache['fft_shape_full'], workers=-1)
				W_fft = fftn(np.flip(self.W, range(self.n_shift_dimensions)), axes=self.shift_dimensions, s=self._cache['fft_shape_full'], workers=-1)
				result_fft = contract('...cn,...cm->...mn', X_fft, W_fft, optimize='optimal')
				result_pad = np.real(ifftn(result_fft, axes=self.shift_dimensions, workers=-1))
				upper_idx = self.H.shape[:self.n_shift_dimensions]
				slices = tuple(slice(upper) for upper in upper_idx)
				result = result_pad[slices]
				return result

			numer = fft_convolve(self.V)
			denum = fft_convolve(self.R)
		else:
			V_padded = self._cache['V_padded']
			R_padded = np.pad(self.R, pad_width=self._cache['pad_width'])
			V_strided = as_strided(V_padded, self._cache['X_strided_W_shape'], self._cache['X_strided_W_strides'], writeable=False)
			R_strided = as_strided(R_padded, self._cache['X_strided_W_shape'], self._cache['X_strided_W_strides'], writeable=False)
			numer = contract(self.W, self._cache['W_labels'], V_strided, self._cache['X_strided_W_labels'], self._cache['H_labels'], optimize='optimal')
			denum = contract(self.W, self._cache['W_labels'], R_strided, self._cache['X_strided_W_labels'], self._cache['H_labels'], optimize='optimal')
		return numer, denum

	def _reconstruction_gradient_W(self) -> np.array:
		"""Positive and negative parts of the gradient of the reconstruction error w.r.t. the dictionary matrix."""
		# TODO: inherit docstring from superclass
		# TODO: computation via FFT
		if self._use_fft:

			def fft_convolve(X):
				X_fft = fftn(X, axes=self.shift_dimensions, s=self._cache['fft_shape_valid'], workers=-1)
				H_fft = fftn(np.flip(self.H, range(self.n_shift_dimensions)), axes=self.shift_dimensions, s=self._cache['fft_shape_valid'], workers=-1)
				result_fft = contract('...cn,...mn->...cm', X_fft, H_fft, optimize='optimal')
				result_pad = np.real(ifftn(result_fft, axes=self.shift_dimensions, workers=-1))
				lower_idx = np.array(self.V.shape[:self.n_shift_dimensions]) - 1
				upper_idx = np.array(self.V.shape[:self.n_shift_dimensions]) + np.array(self.W.shape[:self.n_shift_dimensions]) - 1
				slices = tuple(slice(lower, upper) for lower, upper in zip(lower_idx, upper_idx))
				result = result_pad[slices]
				return result

			numer = fft_convolve(self.V)
			denum = fft_convolve(self.R)
		else:
			H_strided = as_strided(self.H, self._cache['H_strided_V_shape'], self._cache['H_strided_V_strides'], writeable=False)
			numer = np.flip(contract(H_strided, self._cache['H_strided_V_labels'], self.V, self._cache['V_labels'], self._cache['W_labels'], optimize='optimal'), axis=range(self.n_shift_dimensions))
			denum = np.flip(contract(H_strided, self._cache['H_strided_V_labels'], self.R, self._cache['V_labels'], self._cache['W_labels'], optimize='optimal'), axis=range(self.n_shift_dimensions))
		return numer, denum
