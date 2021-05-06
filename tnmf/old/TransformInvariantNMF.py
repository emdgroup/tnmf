"""
Author: Adrian Sosic
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import as_strided
import logging
from scipy.fft import rfftn, irfftn, next_fast_len
from scipy.ndimage import convolve1d
from opt_einsum import contract
from itertools import product
from abc import ABC
from .utils import normalize, shift
from typing import Optional, Tuple
plt.style.use('seaborn')

# TODO: replace 'matrix' with 'tensor' in docstrings
# TODO: indicate 'override' in subclasses


class TransformInvariantNMF(ABC):
	"""Abstract base class for transform-invariant non-negative matrix factorization."""

	def __init__(
			self,
			atom_size: Optional[int],
			n_components: int = 10,
			sparsity_H: float = 0.1,
			refit_H: bool = True,
			n_iterations: int = 100,
			eps: float = 1e-9,
			logger: logging.Logger = None,
			verbose: int = 0,
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
		logger : logging.Logger
			logging.Logger instance used for intermediate output
		verbose : int
			Verbosity level: 0 - show only errors, 1 - include warnings, 2 - include info, 3 - include debug messages
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

		# logger - use default if nothing else is given
		self._logger = logger if logger is not None else logging.getLogger(self.__class__.__name__)
		self._logger.setLevel([logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG][verbose])

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
		self.H = 1 - np.random.random([self.n_transforms, self.n_components, self.n_signals]).astype(self.V.dtype)
		self.W = normalize(1 - np.random.random([self.atom_size, self.n_channels, self.n_components]).astype(self.V.dtype), axis=self._normalization_dims)

	def fit(self, V):
		"""Learns an NMF representation of a given signal matrix."""
		# initialize all matrices
		self.initialize(V)

		# TODO: define stopping criterion
		# iterate the multiplicative update rules
		for i in range(self.n_iterations):
			self._logger.info(f"Iteration: {i}\tReconstruction error: {self.reconstruction_error()}")
			self.update_H()
			self.update_W()

		# TODO: define stopping criterion
		# refit the activations using the learned dictionary
		if self.refit_H:
			self._logger.info("Refitting activations.")
			for i in range(10):
				self.update_H(sparsity=False)

		assert self.H.dtype == self.V.dtype
		assert self.W.dtype == self.V.dtype

		self._logger.info("NMF finished.")

	def reconstruction_error(self) -> float:
		"""Squared error between the input and its reconstruction."""
		return 0.5 * np.sum(np.square(self.V - self.R))

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
		return np.eye(self.n_dim, dtype=self.V.dtype)[None, :, :]


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
		assert np.isreal(V).all()
		super().initialize(V)
		self._init_cache()

	def _init_cache(self):
		"""Caches several fitting related variables."""
		cache = {}

		if self._use_fft:
			# fft shape and functions
			cache['fft_shape'] = [next_fast_len(s) for s in np.array(self.V.shape[:self.n_shift_dimensions]) + np.array(self.H.shape[:self.n_shift_dimensions]) - 1]
			cache['fft_fun'] = lambda x: rfftn(x, axes=self.shift_dimensions, s=cache['fft_shape'], workers=-1)
			cache['ifft_fun'] = lambda x: irfftn(x, axes=self.shift_dimensions, s=cache['fft_shape'], workers=-1)

			# transformed input
			cache['V_fft'] = cache['fft_fun'](self.V)

			# fft details: reconstruction
			lower_idx = np.array(self.W.shape[:self.n_shift_dimensions]) - 1
			upper_idx = np.array(self.V.shape[:self.n_shift_dimensions]) + np.array(self.W.shape[:self.n_shift_dimensions]) - 1
			slices = tuple(slice(lower, upper) for lower, upper in zip(lower_idx, upper_idx))
			cache[('W', 'H')] = {
				'contraction_string': '...cm,...mn->...cn',
				'slices': slices
			}

			# fft details: gradient H computation
			upper_idx = self.H.shape[:self.n_shift_dimensions]
			slices = tuple(slice(upper) for upper in upper_idx)
			cache[('V', 'W')] = {
				'contraction_string': '...cn,...cm->...mn',
				'slices': slices
			}
			cache[('R', 'W')] = cache[('V', 'W')]

			# fft details: gradient W computation
			lower_idx = np.array(self.V.shape[:self.n_shift_dimensions]) - 1
			upper_idx = np.array(self.V.shape[:self.n_shift_dimensions]) + np.array(self.W.shape[:self.n_shift_dimensions]) - 1
			slices = tuple(slice(lower, upper) for lower, upper in zip(lower_idx, upper_idx))
			cache[('V', 'H')] = {
				'contraction_string': '...cn,...mn->...cm',
				'slices': slices
			}
			cache[('R', 'H')] = cache[('V', 'H')]

		else:
			# zero-padding of the signal matrix for full-size correlation
			cache['pad_width'] = (*self.n_shift_dimensions*((self.atom_size-1,)*2,), (0,0), (0,0))
			cache['V_padded'] = np.pad(self.V, pad_width=cache['pad_width'])

			# dimension labels of the data and reconstruction matrices
			cache['V_labels'] = ['d' + str(i) for i in self.shift_dimensions] + ['c', 'n']
			cache['W_labels'] = ['a' + str(i) for i in self.shift_dimensions] + ['c', 'm']
			cache['H_labels'] = ['d' + str(i) for i in self.shift_dimensions] + ['m', 'n']

			# dimension info for striding in gradient_H computation
			cache['X_strided_W_shape'] = (self.atom_size,) * self.n_shift_dimensions + self.H.shape[:-2] + self.V.shape[-2:]
			cache['X_strided_W_labels'] = [s + str(i) for s, i in product(['a', 'd'], self.shift_dimensions)] + ['c', 'n']

			# dimension info for striding in gradient_W computation
			cache['H_strided_V_shape'] = self.V.shape[:self.n_shift_dimensions] + (self.atom_size,) * self.n_shift_dimensions + self.H.shape[-2:]
			cache['H_strided_V_labels'] = [s + str(i) for s, i in product(['d', 'a'], self.shift_dimensions)] + ['m', 'n']

			# dimension info for striding in reconstruction computation
			cache['H_strided_W_shape'] = (self.atom_size,) * self.n_shift_dimensions + self.V.shape[:-2] + self.H.shape[-2:]
			cache['H_strided_W_labels'] = [s + str(i) for s, i in product(['a', 'd'], self.shift_dimensions)] + ['m', 'n']

		self._cache = cache

	def _init_factorization_matrices(self):
		"""Initializes the activation matrix and dictionary matrix."""
		# TODO: inherit docstring from superclass
		self._normalization_dims = self.shift_dimensions
		self.H = 1 - np.random.random((self.n_signals, self.n_components, *self.n_transforms)).astype(self.V.dtype)
		self.H = self.H.transpose((2,3,1,0)).copy()
		self.W = 1 - np.random.random((self.n_components, self.n_channels, *[self.atom_size] * self.n_shift_dimensions)).astype(self.V.dtype)
		self.W = self.W.transpose((2,3,1,0)).copy()
		self.W = normalize(self.W, axis=self._normalization_dims)

	def _gradient_H(self, sparsity: bool = True) -> (np.array, np.array):
		"""Computes the positive and the negative parts of the energy gradient w.r.t. the activation tensor."""
		# TODO: inherit docstring from superclass

		# compute the gradient w.r.t. the reconstruction and sparsity energies
		numer, denum = super()._gradient_H(sparsity)

		# add the inhibition gradient component
		if self.inhibition_range and self.inhibition_strength:
			inhibition = self.H.copy()
			for dim in self.shift_dimensions:
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
		base_array = np.hstack([np.eye(self.atom_size, dtype=self.V.dtype),
								np.zeros([self.atom_size, self.n_dim - self.atom_size], dtype=self.V.dtype)])
		T = np.array([shift(base_array, s).T for s in range(-self.atom_size + 1, self.n_dim)], dtype=self.V.dtype)
		return T


class ImplicitShiftInvariantNMF(BaseShiftInvariantNMF):
	"""Class for shift-invariant non-negative matrix factorization that computes the involved transform operations
	implicitly via correlation/convolution."""

	# TODO: switch original/frequency domains only on demand via setter/getters

	def __init__(self, use_fft=True, **kwargs):
		super().__init__(**kwargs)
		self._use_fft = use_fft
		self._logger.debug(f'Using the {"FFT" if self._use_fft else "non-FFT"} implementation.')

	@property
	def V_fft(self):
		return self._cache['V_fft']

	@property
	def R_fft(self):
		return self._cache['fft_fun'](self.R)

	@property
	def W_fft(self):
		return self._cache['fft_fun'](self.W)

	@property
	def W_reversed_fft(self):
		return self._cache['fft_fun'](np.flip(self.W, axis=self.shift_dimensions))

	@property
	def H_fft(self):
		return self._cache['fft_fun'](self.H)

	@property
	def H_reversed_fft(self):
		return self._cache['fft_fun'](np.flip(self.H, axis=self.shift_dimensions))

	def _fft_convolve(self, arr1, arr2, flip_second=False):
		arr1_fft = getattr(self, arr1 + '_fft')
		if flip_second:
			arr2_fft = getattr(self, arr2 + '_reversed_fft')
		else:
			arr2_fft = getattr(self, arr2 + '_fft')
		result_fft = contract(self._cache[(arr1, arr2)]['contraction_string'], arr1_fft, arr2_fft)
		result_pad = self._cache['ifft_fun'](result_fft)
		result = result_pad[self._cache[(arr1, arr2)]['slices']]
		return result

	def _reconstruct(self) -> np.array:
		"""Reconstructs the signal matrix via (fft-)convolution."""
		if self._use_fft:
			R = self._fft_convolve('W', 'H')
		else:
			H_strided_W_strides = self.H.strides[:self.n_shift_dimensions] + self.H.strides
			H_strided = as_strided(self.H, self._cache['H_strided_W_shape'], H_strided_W_strides, writeable=False)
			R = contract(H_strided, self._cache['H_strided_W_labels'], np.flip(self.W, self.shift_dimensions), self._cache['W_labels'], self._cache['V_labels'], optimize='optimal')
		return R

	def _reconstruction_gradient_H(self) -> np.array:
		"""Positive and negative parts of the gradient of the reconstruction error w.r.t. the activation tensor."""
		# TODO: inherit docstring from superclass
		if self._use_fft:
			numer = self._fft_convolve('V', 'W', flip_second=True)
			denum = self._fft_convolve('R', 'W', flip_second=True)
		else:
			V_padded = self._cache['V_padded']
			V_strided_W_strides = V_padded.strides[:self.n_shift_dimensions] + V_padded.strides
			V_strided = as_strided(V_padded, self._cache['X_strided_W_shape'], V_strided_W_strides, writeable=False)
			numer = contract(self.W, self._cache['W_labels'], V_strided, self._cache['X_strided_W_labels'], self._cache['H_labels'], optimize='optimal')

			R_padded = np.pad(self.R, pad_width=self._cache['pad_width'])
			R_strided_W_strides = R_padded.strides[:self.n_shift_dimensions] + R_padded.strides
			R_strided = as_strided(R_padded, self._cache['X_strided_W_shape'], R_strided_W_strides, writeable=False)
			denum = contract(self.W, self._cache['W_labels'], R_strided, self._cache['X_strided_W_labels'], self._cache['H_labels'], optimize='optimal')
		return numer, denum

	def _reconstruction_gradient_W(self) -> np.array:
		"""Positive and negative parts of the gradient of the reconstruction error w.r.t. the dictionary matrix."""
		# TODO: inherit docstring from superclass
		if self._use_fft:
			numer = self._fft_convolve('V', 'H', flip_second=True)
			denum = self._fft_convolve('R', 'H', flip_second=True)
		else:
			H_strided_V_strides = self.H.strides[:self.n_shift_dimensions] + self.H.strides
			H_strided = as_strided(self.H, self._cache['H_strided_V_shape'], H_strided_V_strides, writeable=False)
			numer = np.flip(contract(H_strided, self._cache['H_strided_V_labels'], self.V, self._cache['V_labels'], self._cache['W_labels'], optimize='optimal'), axis=self.shift_dimensions)
			denum = np.flip(contract(H_strided, self._cache['H_strided_V_labels'], self.R, self._cache['V_labels'], self._cache['W_labels'], optimize='optimal'), axis=self.shift_dimensions)
		return numer, denum
