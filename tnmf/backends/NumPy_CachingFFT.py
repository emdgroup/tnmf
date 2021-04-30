from .Backend import NumpyBackend
import logging
import numpy as np
from scipy.fft import next_fast_len, rfftn, irfftn
from opt_einsum import contract_expression
from typing import Tuple, Optional, List


class CachingFFT(object):
    """
    Wrapper class for conveniently caching and switching back and forth
    between fields in coordinate space and fourier space
    """

    def __init__(
        self,
        field_name: str,
        fft_axes: Optional[Tuple[int]] = None,
        fft_shape: Optional[List[int]] = None, logger: Optional[logging.Logger] = None,
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
        self.c *= other
        return self

    def set_fft_params(self, fft_axes: Tuple[int], fft_shape: List[int]):
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
    def c(self) -> np.array:
        """Getter for field in coordinate space"""
        if self._c is None:
            self._logger.debug(f'Computing {self._field_name}(x) = FFT^-1[ {self._field_name}(f) ]')
            assert self.has_f
            self._c = irfftn(self._f, axes=self._fft_axes, s=self._fft_shape, workers=self._fft_workers)
        return self._c

    @c.setter
    def c(self, c: np.array):
        """Setter for field in coordinate space"""
        self._logger.debug(f'{"Setting" if c is not None else "Clearing"} {self._field_name}(x)')
        self._invalidate()
        self._c = c

    @property
    def f(self) -> np.array:
        """Getter for field in fourier space"""
        if self._f is None:
            self._logger.debug(f'Computing {self._field_name}(f) = FFT[ {self._field_name}(x) ]')
            assert self.has_c()
            self._f = rfftn(self._c, axes=self._fft_axes, s=self._fft_shape, workers=self._fft_workers)
        return self._f

    @f.setter
    def f(self, f: np.array):
        """Setter for field in fourier space"""
        self._logger.debug(f'{"Setting" if f is not None else "Clearing"} {self._field_name}(f)')
        self._invalidate()
        self._f = f

    @property
    def f_reversed(self) -> np.array:
        """Getter for time-reversed field in fourier space, intentionally no setter for now"""
        if self._f_reversed is None:
            self._logger.debug(f'Computing {self._field_name}_rev(f) = FFT[ {self._field_name}(-x) ]')
            assert self.has_c()
            c_reversed = np.flip(self._c, axis=self._fft_axes)
            self._f_reversed = rfftn(c_reversed, axes=self._fft_axes, s=self._fft_shape, workers=self._fft_workers)
        return self._f_reversed


class NumPy_CachingFFT_Backend(NumpyBackend):

    def __init__(
        self,
        logger: logging.Logger = None,
        verbose: int = 0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._logger = logger if logger is not None else logging.getLogger(self.__class__.__name__)
        self._logger.setLevel([logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG][verbose])

    def initialize_matrices(
        self,
        V: np.ndarray,
        atom_shape: Tuple[int, ...],
        n_atoms: int,
        W: Optional[np.array] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:

        w, h = super().initialize_matrices(V, atom_shape, n_atoms, W)

        self._R = CachingFFT('R', logger=self._logger)
        self._V = CachingFFT('V', logger=self._logger)
        self._V.c = V
        sample_shape = V.shape[2:]

        H = CachingFFT('H', logger=self._logger)
        H.c = h

        if W is None:
            W = CachingFFT('W', logger=self._logger)
            W.c = w

        self._cache = {}
        # fft shape and functions
        fft_axes = self._shift_dimensions
        fft_shape = [next_fast_len(s) for s in np.array(sample_shape) + np.array(self._transform_shape) - 1]

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
            'contraction': contract_expression('mc...,nm...->nc...', W.f.shape, H.f.shape),
        }

        # fft details: gradient H computation
        upper_idx = np.array(self._transform_shape)
        self._cache['params_reconstruction_gradient_H'] = {
            'slices': (slice(None, None, 1),) * 2 + tuple(slice(upper) for upper in upper_idx),
            #              sum_c V|R[n, c, ... ] * W[m , c, ...] --> dR / dH[n, m, ...]
            'contraction': contract_expression('nc...,mc...->nm...', self._V.f.shape, W.f_reversed.shape),
        }

        # fft details: gradient W computation
        lower_idx = np.array(sample_shape) - 1
        upper_idx = np.array(sample_shape) + np.array(atom_shape) - 1
        self._cache['params_reconstruction_gradient_W'] = {
            'slices': (slice(None, None, 1),) * 2 + tuple(slice(lower, upper) for lower, upper in zip(lower_idx, upper_idx)),
            #              sum_n V|R[n, c, ... ] * H[n , m, ...]   --> dR / dW[m, c, ...]
            'contraction': contract_expression('nc...,nm...->mc...', self._V.f.shape, H.f_reversed.shape),
        }

        return W, H

    def normalize(self, arr: np.array, axes: Tuple[int]) -> np.array:
        arr.c /= arr.c.sum(axis=axes, keepdims=True)
        return arr

    def _fft_convolve(self, arr1_fft, arr2_fft, contraction, slices):
        result = CachingFFT('fft_convolve', fft_axes=self._cache['fft_axes'], fft_shape=self._cache['fft_shape'])
        result.f = contraction(arr1_fft, arr2_fft)
        return result.c[slices]

    def reconstruction_gradient_W(self, V: np.array, W: np.array, H: np.array) -> Tuple[np.array, np.array]:
        R = self._reconstruct_cachingfft(W, H)
        assert R.c.shape == V.shape
        numer = self._fft_convolve(self._V.f, H.f_reversed, **self._cache['params_reconstruction_gradient_W'])
        denum = self._fft_convolve(self._R.f, H.f_reversed, **self._cache['params_reconstruction_gradient_W'])
        return numer, denum

    def reconstruction_gradient_H(self, V: np.array, W: np.array, H: np.array) -> Tuple[np.array, np.array]:
        R = self._reconstruct_cachingfft(W, H)
        assert R.c.shape == V.shape
        numer = self._fft_convolve(self._V.f, W.f_reversed, **self._cache['params_reconstruction_gradient_H'])
        denum = self._fft_convolve(self._R.f, W.f_reversed, **self._cache['params_reconstruction_gradient_H'])
        return numer, denum

    def _reconstruct_cachingfft(self, W: np.array, H: np.array) -> np.array:
        self._R.c = self._fft_convolve(W.f, H.f, **self._cache['params_reconstruct'])
        return self._R

    def reconstruct(self, W: np.array, H: np.array) -> np.array:
        R = self._reconstruct_cachingfft(W, H)
        return R.c
