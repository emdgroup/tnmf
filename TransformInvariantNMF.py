"""
Author: Adrian Sosic
"""

import logging
import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy.fft import next_fast_len, rfftn, irfftn
from scipy.ndimage import convolve1d
from opt_einsum import contract, contract_expression
from itertools import product
from abc import ABC
from utils import normalize, shift
from CachingFFT import CachingFFT
from typing import Optional, Tuple, Callable, Dict

# TODO: replace 'matrix' with 'tensor' in docstrings
# TODO: indicate 'override' in subclasses
# TODO: refactor fft code parts into functions


def _centered(arr, newshape):
	# Return the center newshape portion of the array.
	newshape = np.asarray(newshape)
	currshape = np.array(arr.shape)
	startind = (currshape - newshape) // 2
	endind = startind + newshape
	myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
	return arr[tuple(myslice)]


def _inputs_swap_needed(mode, shape1, shape2, axes=None):
	if mode != 'valid':
		return False
	if not shape1:
		return False
	if axes is None:
		axes = range(len(shape1))
	ok1 = all(shape1[i] >= shape2[i] for i in axes)
	ok2 = all(shape2[i] >= shape1[i] for i in axes)

	if not (ok1 or ok2):
		raise ValueError("For 'valid' mode, one must be at least as large as the other in every dimension")

	return not ok1


def fftconvolve_sum(in1, in2, mode="full", axes=None, sum_axis=None, padding1=dict(mode='constant', constant_values=0), padding2=dict(mode='constant', constant_values=0)):

	assert in1.ndim == in2.ndim
	if axes is None:
		axes = range(in1.ndim)

	s1 = in1.shape
	s2 = in2.shape
	axes = [a for a in axes if s1[a] != 1 and s2[a] != 1]

	if mode == 'valid':
		if not all(s1[i] >= s2[i] for i in axes):
			in1, in2, s1, s2, padding1, padding2 = in2, in1, s2, s1, padding2, padding1

	shape = [max((s1[i], s2[i])) if i not in axes else s1[i] + s2[i] - 1 for i in range(in1.ndim)]
	fshape = [next_fast_len(shape[a], True) for a in axes]

	pad_width_1 = [((0, fshape[a] - in1.shape[a]) if a in axes else (0, 0)) for a in range(in1.ndim)]
	pad_width_2 = [((0, fshape[a] - in2.shape[a]) if a in axes else (0, 0)) for a in range(in2.ndim)]

	in1_padded = np.pad(in1, pad_width_1, **padding1)
	in2_padded = np.pad(in2, pad_width_2, **padding2)

	sp1 = rfftn(in1_padded, fshape, axes=axes)
	sp2 = rfftn(in2_padded, fshape, axes=axes)
	ret = irfftn(sp1 * sp2, fshape, axes=axes)

	fslice = tuple([slice(sz) for sz in shape])
	ret = ret[fslice]

	if mode == "full":
		ret = ret.copy()
	elif mode == "same":
		ret = _centered(ret, s1).copy()
	elif mode == "valid":
		shape_valid = [ret.shape[a] if a not in axes else s1[a] - s2[a] + 1 for a in range(ret.ndim)]
		ret = _centered(ret, shape_valid).copy()
	else:
		raise ValueError("acceptable mode flags are 'valid', 'same', or 'full'")

	# TODO: the sum could as well be performed in Fourier space, which saves us a number of back/transformations
	return np.sum(ret, axis=sum_axis) if sum_axis is not None else ret


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

		# logger - use default if nothing else is given
		self._logger = logger if logger is not None else logging.getLogger(self.__class__.__name__)
		self._logger.setLevel([logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG][verbose])

		# signal, reconstruction, factorization, and transformation matrices
		self._V = CachingFFT('V', logger=self._logger)
		self._R = CachingFFT('R', logger=self._logger)
		self._T = None
		self._W = CachingFFT('W', logger=self._logger)
		self._H = CachingFFT('H', logger=self._logger)

		# constant to avoid division by zero
		self.eps = eps

		# axis over which the dictionary matrix gets normalized
		self._normalization_dims = 0

	@property
	def V(self) -> np.array:
		"""The signal matrix, set only via fit() """
		return self._V.c

	@property
	def R(self) -> np.array:
		"""The reconstructed signal matrix."""
		if not self._R.has_c():
			self._logger.debug('Reconstructing R')
			self._R.c = self._reconstruct()
		return self._R.c

	@R.setter
	def R(self, R: np.array):
		self._R.c = R

	@property
	def W(self) -> np.array:
		"""The dictionary matrix."""
		return self._W.c

	@W.setter
	def W(self, W: np.array):
		self._W.c = W
		self.R = None

	@property
	def H(self) -> np.array:
		"""The activation tensor."""
		return self._H.c

	@H.setter
	def H(self, H: np.array):
		self._H.c = H
		self.R = None

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
		d: number of input dimensions, i.e. shape of the individual samples
		c: number of input channels
		n: number of input samples
		m: number of basis vectors (dictionary size)
		t: number of basis vector transforms (= 1 for standard NMF without transform invariance), i.e. potential placement positions of the basis vector
		h: number of basis vector dimensions (= d for standard NMF without transform invariance), i.e. atom shape

		Dimensions:
		-----------
		Signal matrix V: 		d x c x n
		Dictionary Matrix W: 	h x c x m
		Activation Tensor H: 	t x m x n
		Transformation Tensor:  t x d x h
		"""
		# store the signal matrix
		self._V.c = np.asarray(V)

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
		self.W = normalize(np.random.random([self.atom_size, self.n_channels, self.n_components]).astype(self.V.dtype), axis=self._normalization_dims)
		self.H = np.random.random([self.n_transforms, self.n_components, self.n_signals]).astype(self.V.dtype)

	def fit(self, V, progress_callback: Callable[['TransformInvariantNMF', int], bool] = None):
		"""Learns an NMF representation of a given signal matrix."""
		# initialize all matrices
		self.initialize(V)

		# TODO: define stopping criterion
		# iterate the multiplicative update rules
		for i in range(self.n_iterations):
			if progress_callback is not None:
				if not progress_callback(self, i):
					break
			else:
				self._logger.info(f"Iteration: {i}\tCost function: {self.cost_function()}")

			self.update_H()
			self.update_W()

		# TODO: define stopping criterion
		# refit the activations using the learned dictionary
		if self.refit_H:
			self._logger.info("Refitting activations.")
			for i in range(10):
				self.update_H(sparsity=False)

				if progress_callback is not None:
					progress_callback(self, -i-1)  # negative iteration numbers indicate refitting
				else:
					self._logger.info(f"Refit iteration: {i}\tCost function: {self.cost_function()}")

		assert self.H.dtype == self.V.dtype
		assert self.W.dtype == self.V.dtype

		self._logger.info("NMF finished.")

	def cost_term_reconstruction(self) -> float:
		"""L2 norm error between the input and its reconstruction."""
		difference = (self.V - self.R).ravel()
		return 0.5 * np.dot(difference, difference)

	def cost_term_sparsity(self) -> float:
		"""Cost function contribution from the sparsity constraint for the activation tensor"""
		return self.sparsity * self.H.sum()

	def cost_term_inhibition(self) -> float:
		"""Cost function contribution from the inhibition term"""
		return self._inhibition_cost

	def cost_function(self) -> Dict[str, float]:
		"""Returns the individual contributions and the total value of the cost function"""
		cost = {
			'reconstruction':  self.cost_term_reconstruction(),
			'sparsity':  self.cost_term_sparsity(),
			'inhibition':  self.cost_term_inhibition(),
			}
		cost['total'] = sum(cost.values())
		return cost


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

	def __init__(self, reconstruction_mode: str = 'full', inhibition_range: Optional[int] = None, inhibition_strength: float = 0.1, **kwargs):
		"""
		Parameters
		----------
		inhibition_range : int
			Number of neighboring activation elements in each direction that exert an inhibitory effect.
			If 'None', the range is set to the minimal range that covers the size of a dictionary element.
		"""
		# set the basic parameters
		super().__init__(**kwargs)

		self._mode_R = reconstruction_mode
		self._mode_H = {'full': 'valid', 'valid': 'full', 'same': 'same', }[reconstruction_mode]
		assert self._mode_R != 'same' or self.atom_size % 2 == 1  # for 'same', we can only understand the overall situation for odd atom sizes for now
		
		# default inhibition range = minimal range to cover the atom size
		if inhibition_range is None:
			inhibition_range = int(np.ceil(self.atom_size / 2))

		# store the inhibition parameters and construct the inhibition kernel
		self.inhibition_range = inhibition_range
		self.inhibition_strength = inhibition_strength
		self.kernel = 1 - ((np.arange(-inhibition_range, inhibition_range + 1) / inhibition_range) ** 2)
		self._inhibition_cost = 0.

	@property
	def n_dim(self) -> Tuple[int]:
		"""Number of input dimensions."""
		return tuple(self.V.shape[:-2])

	@property
	def n_transforms(self) -> Tuple[int]:
		"""Number of dictionary transforms."""
		# TODO: inherit docstring from superclass
		if self._mode_R == 'valid':
			return tuple(np.array(self.n_dim) + self.atom_size - 1)
		elif self._mode_R == 'full':
			return tuple(np.array(self.n_dim) - self.atom_size + 1)
		elif self._mode_R == 'same':
			return tuple(np.array(self.n_dim))
		else:
			raise ValueError

	@property
	def n_shift_dimensions(self):
		"""The number of shift invariant input dimensions."""
		return self.V.ndim - 2

	@property
	def shift_dimensions(self) -> Tuple[int]:
		"""The dimension index of the shift invariant input dimensions."""
		return tuple(range(self.n_shift_dimensions))

	def initialize(self, V):
		assert np.isreal(V).all()
		super().initialize(V)
		self._normalization_dims = self.shift_dimensions

	def _init_factorization_matrices(self):
		"""Initializes the activation matrix and dictionary matrix."""
		# TODO: inherit docstring from superclass
		self.W = normalize(np.random.random([*[self.atom_size] * self.n_shift_dimensions, self.n_channels, self.n_components]).astype(self.V.dtype), axis=self._normalization_dims)
		self.H = np.random.random([*self.n_transforms, self.n_components, self.n_signals]).astype(self.V.dtype)

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
			inhibition = self.inhibition_strength * (inhibition - self.H)
			denum = denum + inhibition
			self._inhibition_cost = np.sum(self.H * inhibition)

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

	def __init__(self, method='cachingFFT', input_padding=dict(mode='constant', constant_values=0), **kwargs):
		super().__init__(**kwargs)
		assert method in ('cachingFFT', 'contract', 'fftconvolve')
		assert self._mode_R == 'valid' or method == 'fftconvolve'  # only fftconvolve support the different modes properly
		self._method = method
		self._input_padding = input_padding
		self._logger.debug(f'Using method {self._method}.')
		self._cache = {}

	def initialize(self, V):
		super().initialize(V)
		self._init_cache()

	def _init_cache(self):
		"""Caches several fitting related variables."""
		cache = {}

		if self._method == 'cachingFFT':
			# fft shape and functions
			fft_axes = self.shift_dimensions
			fft_shape = [next_fast_len(s) for s in np.array(self.V.shape[:self.n_shift_dimensions]) + np.array(self.H.shape[:self.n_shift_dimensions]) - 1]

			cache['fft_axes'] = fft_axes
			cache['fft_shape'] = fft_shape
			self._V.set_fft_params(fft_axes, fft_shape)
			self._R.set_fft_params(fft_axes, fft_shape)
			self._W.set_fft_params(fft_axes, fft_shape)
			self._H.set_fft_params(fft_axes, fft_shape)

			# fft details: reconstruction
			lower_idx = np.array(self.W.shape[:self.n_shift_dimensions]) - 1
			upper_idx = np.array(self.V.shape[:self.n_shift_dimensions]) + np.array(self.W.shape[:self.n_shift_dimensions]) - 1
			cache['params_reconstruct'] = {
				'slices': tuple(slice(lower, upper) for lower, upper in zip(lower_idx, upper_idx)),
				'contraction': contract_expression('...cm,...mn->...cn', self.W_fft.shape, self.H_fft.shape),
			}

			# fft details: gradient H computation
			upper_idx = self.H.shape[:self.n_shift_dimensions]
			cache['params_reconstruction_gradient_H'] = {
				'slices': tuple(slice(upper) for upper in upper_idx),
				'contraction': contract_expression('...cn,...cm->...mn', self.V_fft.shape, self.W_reversed_fft.shape),
			}

			# fft details: gradient W computation
			lower_idx = np.array(self.V.shape[:self.n_shift_dimensions]) - 1
			upper_idx = np.array(self.V.shape[:self.n_shift_dimensions]) + np.array(self.W.shape[:self.n_shift_dimensions]) - 1
			cache['params_reconstruction_gradient_W'] = {
				'slices': tuple(slice(lower, upper) for lower, upper in zip(lower_idx, upper_idx)),
				'contraction': contract_expression('...cn,...mn->...cm', self.V_fft.shape, self.H_reversed_fft.shape),
			}
		elif self._method == 'contract':
			# zero-padding of the signal matrix for full-size correlation
			cache['pad_width'] = (*self.n_shift_dimensions * ((self.atom_size - 1,) * 2,), (0, 0), (0, 0))
			cache['V_padded'] = np.pad(self.V, pad_width=cache['pad_width'])

			# dimension labels of the data and reconstruction matrices
			cache['V_labels'] = ['d' + str(i) for i in self.shift_dimensions] + ['c', 'n']
			cache['W_labels'] = ['a' + str(i) for i in self.shift_dimensions] + ['c', 'm']
			cache['H_labels'] = ['d' + str(i) for i in self.shift_dimensions] + ['m', 'n']

			# dimension info for striding in gradient_H computation
			cache['X_strided_W_shape'] = (self.atom_size,) * self.n_shift_dimensions + self.H.shape[:-2] + self.V.shape[-2:]
			cache['X_strided_W_strides'] = cache['V_padded'].strides[:self.n_shift_dimensions] + cache['V_padded'].strides
			cache['X_strided_W_labels'] = [s + str(i) for s, i in product(['a', 'd'], self.shift_dimensions)] + ['c', 'n']

			# dimension info for striding in gradient_W computation
			cache['H_strided_V_shape'] = self.V.shape[:self.n_shift_dimensions] + (self.atom_size,) * self.n_shift_dimensions + self.H.shape[-2:]
			cache['H_strided_V_strides'] = self.H.strides[:self.n_shift_dimensions] + self.H.strides
			cache['H_strided_V_labels'] = [s + str(i) for s, i in product(['d', 'a'], self.shift_dimensions)] + ['m', 'n']

			# dimension info for striding in reconstruction computation
			cache['H_strided_W_shape'] = (self.atom_size,) * self.n_shift_dimensions + self.V.shape[:-2] + self.H.shape[-2:]
			cache['H_strided_W_strides'] = self.H.strides[:self.n_shift_dimensions] + self.H.strides
			cache['H_strided_W_labels'] = [s + str(i) for s, i in product(['a', 'd'], self.shift_dimensions)] + ['m', 'n']
		elif self._method == 'fftconvolve':
			pass
		else:
			raise ValueError('Unsupported method.')

		self._cache = cache

	@property
	def V_fft(self) -> np.array:
		assert self._method == 'cachingFFT'
		return self._V.f

	@property
	def R_fft(self) -> np.array:
		assert self._method == 'cachingFFT'
		# make sure that R.c is up-to-date
		_ = self.R
		return self._R.f

	@property
	def W_fft(self) -> np.array:
		assert self._method == 'cachingFFT'
		return self._W.f

	@property
	def W_reversed_fft(self) -> np.array:
		assert self._method == 'cachingFFT'
		return self._W.f_reversed

	@property
	def H_fft(self) -> np.array:
		assert self._method == 'cachingFFT'
		return self._H.f

	@property
	def H_reversed_fft(self) -> np.array:
		assert self._method == 'cachingFFT'
		return self._H.f_reversed

	def _fft_convolve(self, arr1_fft, arr2_fft, contraction, slices):
		result = CachingFFT('fft_convolve', fft_axes=self._cache['fft_axes'], fft_shape=self._cache['fft_shape'])
		result.f = contraction(arr1_fft, arr2_fft)
		return result.c[slices]

	def _reconstruct(self) -> np.array:
		"""Reconstructs the signal matrix via (fft-)convolution."""
		if self._method == 'cachingFFT':
			R = self._fft_convolve(self.W_fft, self.H_fft, **self._cache['params_reconstruct'])
		elif self._method == 'contract':
			H_strided = as_strided(self.H, self._cache['H_strided_W_shape'], self._cache['H_strided_W_strides'], writeable=False)
			R = contract(H_strided, self._cache['H_strided_W_labels'], np.flip(self.W, self.shift_dimensions), self._cache['W_labels'], self._cache['V_labels'], optimize='optimal')
		elif self._method == 'fftconvolve':
			R = fftconvolve_sum(self.H[...,np.newaxis,:,:], self.W[...,:,np.newaxis], padding1=self._input_padding, mode=self._mode_R, axes=self.shift_dimensions, sum_axis=-2)
		else:
			R = None
		return R

	def partial_reconstruct(self, sample: int, channel: int, atom: int) -> np.array:
		"""Reconstructs a sample/channel via (fft-)convolution using only a specified atom."""
		if self._method == 'cachingFFT':
			R = self._fft_convolve(self.W_fft[..., channel:channel+1, atom:atom+1],
								   self.H_fft[..., atom:atom+1, sample:sample+1], **self._cache['params_reconstruct'])
		elif self._method == 'contract':
			H_strided = as_strided(self.H[..., atom:atom+1, sample:sample+1], self._cache['H_strided_W_shape'], self._cache['H_strided_W_strides'], writeable=False)
			R = contract(H_strided, self._cache['H_strided_W_labels'], np.flip(self.W[..., channel:channel+1, atom:atom+1], self.shift_dimensions), self._cache['W_labels'], self._cache['V_labels'], optimize='optimal')
		elif self._method == 'fftconvolve':
			R = fftconvolve_sum(self.H[..., atom:atom+1, sample:sample+1], self.W[..., channel:channel+1, atom:atom+1], padding1=self._input_padding, mode=self._mode_R, axes=self.shift_dimensions)
		else:
			assert False
			R = None
		return R

	def _reconstruction_gradient_H(self) -> np.array:
		"""Positive and negative parts of the gradient of the reconstruction error w.r.t. the activation tensor."""
		# TODO: inherit docstring from superclass
		if self._method == 'cachingFFT':
			numer = self._fft_convolve(self.V_fft, self.W_reversed_fft, **self._cache['params_reconstruction_gradient_H'])
			denum = self._fft_convolve(self.R_fft, self.W_reversed_fft, **self._cache['params_reconstruction_gradient_H'])
		elif self._method == 'contract':
			V_padded = self._cache['V_padded']
			R_padded = np.pad(self.R, pad_width=self._cache['pad_width'])
			V_strided = as_strided(V_padded, self._cache['X_strided_W_shape'], self._cache['X_strided_W_strides'], writeable=False)
			R_strided = as_strided(R_padded, self._cache['X_strided_W_shape'], self._cache['X_strided_W_strides'], writeable=False)
			numer = contract(self.W, self._cache['W_labels'], V_strided, self._cache['X_strided_W_labels'], self._cache['H_labels'], optimize='optimal')
			denum = contract(self.W, self._cache['W_labels'], R_strided, self._cache['X_strided_W_labels'], self._cache['H_labels'], optimize='optimal')
		elif self._method == 'fftconvolve':
			reverse = (slice(None, None, -1),) * self.n_shift_dimensions
			W_reversed = self.W[reverse]
			numer = fftconvolve_sum(self.V[...,:,np.newaxis,:], W_reversed[...,:,:,np.newaxis], padding1=self._input_padding, mode=self._mode_H, axes=self.shift_dimensions, sum_axis=-3)
			denum = fftconvolve_sum(self.R[...,:,np.newaxis,:], W_reversed[...,:,:,np.newaxis], padding1=self._input_padding, mode=self._mode_H, axes=self.shift_dimensions, sum_axis=-3)
		else:
			assert False
			numer, denum = None, None
		return numer, denum

	def _reconstruction_gradient_W(self) -> np.array:
		"""Positive and negative parts of the gradient of the reconstruction error w.r.t. the dictionary matrix."""
		# TODO: inherit docstring from superclass
		# TODO: The resulting numer and denum are of size h x c x m, but the fourier space computation creates a d x c x m matrix, which is much larger.
		#       Maybe, this should be done without Fourier transforms just as a direct convolution to make it even faster??
		if self._method == 'cachingFFT':
			numer = self._fft_convolve(self.V_fft, self.H_reversed_fft, **self._cache['params_reconstruction_gradient_W'])
			denum = self._fft_convolve(self.R_fft, self.H_reversed_fft, **self._cache['params_reconstruction_gradient_W'])
		elif self._method == 'contract':
			H_strided = as_strided(self.H, self._cache['H_strided_V_shape'], self._cache['H_strided_V_strides'], writeable=False)
			numer = np.flip(contract(H_strided, self._cache['H_strided_V_labels'], self.V, self._cache['V_labels'], self._cache['W_labels'], optimize='optimal'), axis=self.shift_dimensions)
			denum = np.flip(contract(H_strided, self._cache['H_strided_V_labels'], self.R, self._cache['V_labels'], self._cache['W_labels'], optimize='optimal'), axis=self.shift_dimensions)
		elif self._method == 'fftconvolve':
			reverse = (slice(None, None, -1),) * self.n_shift_dimensions
			H_reversed = self.H[reverse]
			numer = fftconvolve_sum(self.V[:,:,:,np.newaxis,:], H_reversed[:,:,np.newaxis,:,:], padding1=self._input_padding, padding2=self._input_padding, mode='valid', axes=self.shift_dimensions, sum_axis=-1)
			denum = fftconvolve_sum(self.R[:,:,:,np.newaxis,:], H_reversed[:,:,np.newaxis,:,:], padding1=self._input_padding, padding2=self._input_padding, mode='valid', axes=self.shift_dimensions, sum_axis=-1)
		else:
			assert False
			numer, denum = None, None
		return numer, denum
