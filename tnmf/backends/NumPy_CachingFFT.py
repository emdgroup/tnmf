# TODO: consider adding shape getters to CachingFFT
# TODO: this backend has a logger member but the other backends don't

import logging
from typing import Tuple, Optional, Union
from copy import copy

import numpy as np
from scipy.fft import next_fast_len, rfftn, irfftn
from scipy.ndimage import convolve1d
from opt_einsum import contract_expression

from ._NumPyBackend import NumPyBackend


class CachingFFT():
    """
    Wrapper class for conveniently caching and switching back and forth
    between fields in coordinate space and fourier space
    """

    def __init__(
        self,
        field_name: str,
        fft_axes: Optional[Tuple[int, ...]] = None,
        fft_shape: Optional[Tuple[int, ...]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self._c = None  # field in coordinate space
        self._f = None  # field in fourier space
        self._f_reversed = None  # time-reversed field in fourier space
        self._fft_axes = fft_axes
        self._fft_shape = fft_shape
        self._fft_workers = -1
        self._logger = logger if logger is not None else logging.getLogger(self.__class__.__name__)
        self._field_name = field_name

    def __imul__(self, other):
        self.c *= other.c if isinstance(other, CachingFFT) else other
        return self

    def __isub__(self, other):
        self.c -= other.c if isinstance(other, CachingFFT) else other
        return self

    def __iadd__(self, other):
        self.c += other.c if isinstance(other, CachingFFT) else other
        return self

    def __itruediv__(self, other):
        self.c /= other.c if isinstance(other, CachingFFT) else other
        return self

    def set_fft_params(self, fft_axes: Tuple[int, ...], fft_shape: Tuple[int, ...]):
        self._fft_axes = fft_axes
        self._fft_shape = fft_shape

    def _invalidate(self):
        self._c = None
        self._f = None
        self._f_reversed = None

    def has_c(self) -> bool:
        """Check if the field in coordinate space has already been computed"""
        return self._c is not None

    def has_f(self) -> bool:
        """Check if the field in fourier space has already been computed"""
        return self._f is not None

    @property
    def c(self) -> np.ndarray:
        """Getter for field in coordinate space"""
        if self._c is None:
            self._logger.debug(f'Computing {self._field_name}(x) = FFT^-1[ {self._field_name}(f) ]', )
            assert self.has_f
            self._c = irfftn(self._f, axes=self._fft_axes, s=self._fft_shape, workers=self._fft_workers)
        return self._c

    @c.setter
    def c(self, c: np.ndarray):
        """Setter for field in coordinate space"""
        self._logger.debug(f'{"Setting" if c is not None else "Clearing"} {self._field_name}(x)')
        self._invalidate()
        self._c = c

    @property
    def f(self) -> np.ndarray:
        """Getter for field in fourier space"""
        if self._f is None:
            self._logger.debug(f'Computing {self._field_name}(f) = FFT[ {self._field_name}(x) ]')
            assert self.has_c()
            self._f = rfftn(self._c, axes=self._fft_axes, s=self._fft_shape, workers=self._fft_workers)
        return self._f

    @f.setter
    def f(self, f: np.ndarray):
        """Setter for field in fourier space"""
        self._logger.debug(f'{"Setting" if f is not None else "Clearing"} {self._field_name}(f)')
        self._invalidate()
        self._f = f

    @property
    def f_reversed(self) -> np.ndarray:
        """Getter for time-reversed field in fourier space, intentionally no setter for now"""
        if self._f_reversed is None:
            self._logger.debug(f'Computing {self._field_name}_rev(f) = FFT[ {self._field_name}(-x) ]')
            assert self.has_c()
            c_reversed = np.flip(self._c, axis=self._fft_axes)
            self._f_reversed = rfftn(c_reversed, axes=self._fft_axes, s=self._fft_shape, workers=self._fft_workers)
        return self._f_reversed


class NumPy_CachingFFT_Backend(NumPyBackend):

    def __init__(
        self,
        logger: logging.Logger = None,
        verbose: int = 0,
        reconstruction_mode: str = 'valid',
    ):
        if reconstruction_mode != 'valid':
            raise NotImplementedError
        super().__init__(reconstruction_mode=reconstruction_mode)
        self._logger = logger if logger is not None else logging.getLogger(self.__class__.__name__)
        self._logger.setLevel([logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG][verbose])
        self._V = None
        self._R = None
        self._R_partial = None
        self._cache = {}

    def _initialize_matrices(
        self,
        V: np.ndarray,
        atom_shape: Tuple[int, ...],
        n_atoms: int,
        W: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:

        w, h = super()._initialize_matrices(V, atom_shape, n_atoms, W)

        self._R = CachingFFT('R', logger=self._logger)
        self._R_partial = CachingFFT('R_partial', logger=self._logger)
        self._V = CachingFFT('V', logger=self._logger)
        self._V.c = V
        sample_shape = V.shape[2:]

        H = CachingFFT('H', logger=self._logger)
        H.c = h

        if W is None:
            W = CachingFFT('W', logger=self._logger)
            W.c = w

        # fft shape and functions
        fft_axes = self._shift_dimensions
        fft_shape = tuple(next_fast_len(s) for s in np.array(sample_shape) + np.array(self._transform_shape) - 1)

        self._cache['fft_axes'] = fft_axes
        self._cache['fft_shape'] = fft_shape

        self._V.set_fft_params(fft_axes, fft_shape)
        self._R.set_fft_params(fft_axes, fft_shape)
        W.set_fft_params(fft_axes, fft_shape)
        H.set_fft_params(fft_axes, fft_shape)

        # fft details: reconstruction
        lower_idx = np.array(atom_shape) - 1
        upper_idx = np.array(sample_shape) + np.array(atom_shape) - 1
        self._cache['params_reconstruct'] = {
            'slices': (slice(None, None, 1),) * 2 + tuple(slice(lower, upper) for lower, upper in zip(lower_idx, upper_idx)),
            #              sum_m W[m, c, ...] * H[n, m, ... ] --> R[n, c, ...]
            'contraction': contract_expression('mc...,nm...->nc...',
                                               W.f.shape,
                                               H.f.shape),
        }

        # fft details: gradient H computation
        upper_idx = np.array(self._transform_shape)
        self._cache['params_reconstruction_gradient_H'] = {
            'slices': (slice(None, None, 1),) * 2 + tuple(slice(upper) for upper in upper_idx),
            #              sum_c V|R[n, c, ... ] * W[m , c, ...] --> dR / dH[n, m, ...]
            'contraction': contract_expression('nc...,mc...->nm...',
                                               self._V.f.shape,
                                               W.f_reversed.shape),
        }

        # fft details: gradient W computation
        lower_idx = np.array(sample_shape) - 1
        upper_idx = np.array(sample_shape) + np.array(atom_shape) - 1
        # TODO: understand why pylint triggers a warning for H.f_reversed.shape but not for W.f_reversed or H.f
        self._cache['params_reconstruction_gradient_W'] = {
            'slices': (slice(None, None, 1),) * 2 + tuple(slice(lower, upper) for lower, upper in zip(lower_idx, upper_idx)),
            #              sum_n V|R[n, c, ... ] * H[n , m, ...]   --> dR / dW[m, c, ...]
            'contraction': contract_expression('nc...,nm...->mc...',
                                               self._V.f.shape,
                                               H.f_reversed.shape),  # pylint: disable=no-member
        }

        return W, H

    @staticmethod
    def to_ndarray(arr: CachingFFT) -> np.ndarray:
        return arr.c

    @staticmethod
    def normalize(arr: CachingFFT, axis: Optional[Union[int, Tuple[int, ...]]] = None):
        # TODO: overwriting the parent method can be avoided by redefining the division operator of CachingFFT and defining
        #   a common "array type" that can handle both np.ndarray and CachingFFT objects
        arr.c /= arr.c.sum(axis=axis, keepdims=True)

    @staticmethod
    def convolve_multi_1d(arr: CachingFFT, kernels: Tuple[np.ndarray, ...], axes: Tuple[int, ...]) -> np.ndarray:
        assert len(kernels) == len(axes)

        convolved = copy(arr)
        for a, kernel in zip(axes, kernels):
            # TODO: it should be possible to formulate this in Fourier space
            convolved.c = convolve1d(convolved.c, kernel, axis=a, mode='constant', cval=0.0)

        return convolved

    def _fft_convolve(self, arr1_fft, arr2_fft, contraction, slices):
        result = CachingFFT('fft_convolve', fft_axes=self._cache['fft_axes'], fft_shape=self._cache['fft_shape'])
        result.f = contraction(arr1_fft, arr2_fft)
        result.c = result.c[slices]
        return result

    def reconstruction_gradient_W(self, V: np.ndarray, W: np.ndarray, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        R = self.reconstruct(W, H)
        assert R.c.shape == V.shape
        neg = self._fft_convolve(self._V.f, H.f_reversed, **self._cache['params_reconstruction_gradient_W'])
        pos = self._fft_convolve(self._R.f, H.f_reversed, **self._cache['params_reconstruction_gradient_W'])
        return neg, pos

    def reconstruction_gradient_H(self, V: np.ndarray, W: np.ndarray, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        R = self.reconstruct(W, H)
        assert R.c.shape == V.shape
        neg = self._fft_convolve(self._V.f, W.f_reversed, **self._cache['params_reconstruction_gradient_H'])
        pos = self._fft_convolve(self._R.f, W.f_reversed, **self._cache['params_reconstruction_gradient_H'])
        return neg, pos

    def reconstruct(self, W: np.ndarray, H: np.ndarray) -> np.ndarray:
        self._R = self._fft_convolve(W.f, H.f, **self._cache['params_reconstruct'])
        return self._R

    def partial_reconstruct(self, W: np.ndarray, H: np.ndarray, i_atom: int) -> np.ndarray:
        self._R_partial.c = self._fft_convolve(
            W.f[i_atom:i_atom+1], H.f[:, i_atom:i_atom+1], **self._cache['params_reconstruct'])
        return self._R_partial
