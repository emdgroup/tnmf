"""
A module that provides a NumPy based backend for computing the gradients of the factorization model.
Shift-invariance is implemented via fast convolution in the Fourier domain using :func:`scipy.fft.rfftn`
and :func:`scipy.fft.irfftn` with additional caching of the Fourier transformed arrays compared to
:mod:`tnmf.backends.NumPy_FFT`.
"""
# TODO: consider adding shape getters to CachingFFT
# TODO: this backend has a logger member but the other backends don't

import logging
from typing import Dict, Tuple, Optional, Union
from copy import copy

import numpy as np
from opt_einsum import contract_expression
from opt_einsum.contract import ContractExpression
from scipy.fft import next_fast_len, rfftn, irfftn
from scipy.ndimage import convolve1d

from ._NumPyFFTBackend import NumPyFFTBackend


class CachingFFT:
    """
    Wrapper class for conveniently caching and switching back and forth between arrays in coordinate space and their
    representations in Fourier space.
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
        self._f_padded = None  # fourier transform of padded field
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
        self._f_padded = None
        self._f_reversed = None

    def has_c(self) -> bool:
        """Check if the field in coordinate space has already been computed"""
        return self._c is not None

    def has_f(self) -> bool:
        """Check if the field in fourier space has already been computed"""
        return self._f is not None

    @property
    def shape(self) -> Tuple[int]:
        return self._c.shape

    @property
    def ndim(self) -> int:
        return self._c.ndim

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

    def f_padded(self, pad_mode: Dict = None, pad_width: Tuple[Tuple[int, int], ...] = None,) -> np.ndarray:
        """Getter for padded field in fourier space"""
        if self._f_padded is None:
            self._logger.debug(f'Computing {self._field_name}_padded(f) = FFT[ {self._field_name}_padded(x) ]')
            assert self.has_c()

            c = np.pad(self._c, pad_width, **pad_mode)
            # TODO: we should actually make sure that the padding does not change between calls
            self._f_padded = rfftn(c, axes=self._fft_axes, s=self._fft_shape, workers=self._fft_workers)
        return self._f_padded

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


class NumPy_CachingFFT_Backend(NumPyFFTBackend):
    r"""
    A NumPy based backend that performs convolutions and contractions for computing the gradients of the factorization model
    via FFT, similar to :class:`.NumPy_FFT_Backend`. However, the Fourier representations of the associated arrays are cached
    in order to reduce the number of Fourier transformations involved to a minimum.
    """
    def __init__(
        self,
        logger: logging.Logger = None,
        verbose: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._logger = logger if logger is not None else logging.getLogger(self.__class__.__name__)
        self._logger.setLevel([logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG][verbose])
        self._V = None

    def _initialize_matrices(
        self,
        V: np.ndarray,
        atom_shape: Tuple[int, ...],
        n_atoms: int,
        W: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:

        w, h = super()._initialize_matrices(V, atom_shape, n_atoms, W)

        # fft shape and functions
        fft_axes = self._shift_dimensions
        fft_shape = np.array(self._sample_shape) + np.array(self._transform_shape) - 1
        if self._pad_mode is not None:
            fft_shape += np.asarray(self._padding_right).sum(axis=1)
        fft_shape = tuple(next_fast_len(s) for s in np.array(self._sample_shape) + np.array(self._transform_shape) - 1)

        self._V = CachingFFT('V', fft_axes=fft_axes, fft_shape=fft_shape, logger=self._logger)
        self._V.c = V

        H = CachingFFT('H', fft_axes=fft_axes, fft_shape=fft_shape, logger=self._logger)
        H.c = h

        if W is None:
            W = CachingFFT('W', fft_axes=fft_axes, fft_shape=fft_shape, logger=self._logger)
            W.c = w

        # add unpadded and unsliced axes and the necessary fft_shape param
        unpadded = ((0, 0), ) * (V.ndim - len(self._shift_dimensions))
        unsliced = (slice(None), ) * (V.ndim - len(self._shift_dimensions))
        for key in self.fft_params:
            self.fft_params[key]['pad_width'] = unpadded + self.fft_params[key]['pad_width']
            self.fft_params[key]['slices'] = unsliced + self.fft_params[key]['slices']
            self.fft_params[key]['fft_shape'] = fft_shape

        # sum_c V|R[n, c, ... ] * W[m , c, ...] --> dR / dH[n, m, ...]
        self.fft_params['reconstruct']['contraction'] = contract_expression(
            'nm...,mc...->nc...', H.f.shape, W.f.shape)

        # sum_c V|R[n, c, ... ] * W[m , c, ...] --> dR / dH[n, m, ...]
        self.fft_params['grad_H']['contraction'] = contract_expression(
            'nc...,mc...->nm...', self._V.f.shape, W.f_reversed.shape)

        # sum_n V|R[n, c, ... ] * H[n, m, ...]   --> dR / dW[m, c, ...]
        self.fft_params['grad_W']['contraction'] = contract_expression(  # TODO: why is the pylint annotation necessary?
            'nc...,nm...->mc...', self._V.f.shape, H.f_reversed.shape)   # pylint: disable=no-member

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
    def convolve_multi_1d(arr: CachingFFT, kernels: Tuple[np.ndarray, ...], axes: Tuple[int, ...]) -> CachingFFT:
        assert len(kernels) == len(axes)

        convolved = copy(arr)
        for a, kernel in zip(axes, kernels):
            # TODO: it should be possible to formulate this in Fourier space
            convolved.c = convolve1d(convolved.c, kernel, axis=a, mode='constant', cval=0.0)

        return convolved

    @staticmethod
    def _fft_convolve(
        name: Tuple[str, ...],
        arr1: Tuple[np.ndarray, ...],
        arr2: np.ndarray,
        contraction: ContractExpression,
        slices: Tuple[slice, ...],
        fft_axes: Tuple[int, ...],
        fft_shape: Tuple[int, ...],
        pad_mode: Dict = None,
        pad_width: Tuple[Tuple[int, int], ...] = None,
        correlate: bool = False,
        arr1_slice: Tuple[slice, ...] = None,
        arr2_slice: Tuple[slice, ...] = None,
    ) -> CachingFFT:
        arr1_fft = (a.f if pad_mode is None else a.f_padded(pad_mode, pad_width) for a in arr1)
        arr2_fft = arr2.f_reversed if correlate else arr2.f

        arr1_slice = slice(None) if arr1_slice is None else arr1_slice
        arr2_slice = slice(None) if arr2_slice is None else arr2_slice

        ret = tuple()
        for a, n in zip(arr1_fft, name):
            result = CachingFFT(n, fft_axes=fft_axes, fft_shape=fft_shape)
            result.f = contraction(a[arr1_slice], arr2_fft[arr2_slice])
            result.c = result.c[slices]
            ret += (result, )
        return ret

    def reconstruction_gradient_W(self, V: np.ndarray, W: CachingFFT, H: CachingFFT) -> Tuple[CachingFFT, CachingFFT]:
        R = self.reconstruct(W, H)
        assert R.c.shape == V.shape
        return self._fft_convolve(('neg_W', 'pos_W'), (self._V, R), H, **self.fft_params['grad_W'])

    def reconstruction_gradient_H(self, V: np.ndarray, W: CachingFFT, H: CachingFFT) -> Tuple[CachingFFT, CachingFFT]:
        R = self.reconstruct(W, H)
        assert R.c.shape == V.shape
        return self._fft_convolve(('neg_H', 'pos_H'), (self._V, R), W, **self.fft_params['grad_H'])

    def reconstruct(self, W: CachingFFT, H: CachingFFT) -> CachingFFT:
        R, = self._fft_convolve(('R', ), (H, ), W, **self.fft_params['reconstruct'])
        return R

    def partial_reconstruct(self, W: np.ndarray, H: np.ndarray, i_atom: int) -> np.ndarray:
        R_partial, = self._fft_convolve(
            ('R_partial', ), (H, ), W, **self.fft_params['reconstruct'],
            arr1_slice=(slice(None), slice(i_atom, i_atom+1)),
            arr2_slice=(slice(i_atom, i_atom+1)),
            )
        return R_partial
