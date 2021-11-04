"""
A module that provides a NumPy based backend for computing the gradients of the factorization model.
Shift-invariance is implemented via fast convolution in the Fourier domain using :func:`scipy.fft.rfftn`
and :func:`scipy.fft.irfftn` with additional caching of the Fourier transformed arrays compared to
:mod:`tnmf.backends.NumPy_FFT`.
"""
# TODO: this backend has a logger member but the other backends don't

import logging
from typing import Dict, Tuple, Optional, Union
from copy import copy

import numpy as np
from opt_einsum.contract import ContractExpression
from scipy.fft import rfftn, irfftn
from scipy.ndimage import convolve1d

from ._Backend import sliceNone
from ._NumPyFFTBackend import NumPyFFTBackend


class CachingFFT:
    """
    Wrapper class for conveniently caching and switching back and forth between arrays in coordinate space and their
    representations in Fourier space.
    """
    def __init__(
        self,
        field_name: str,
        c: Optional[np.ndarray] = None,
        fft_axes: Optional[Tuple[int, ...]] = None,
        fft_shape: Optional[Tuple[int, ...]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self._c: np.ndarray = c  # field in coordinate space
        self._f: np.ndarray = None  # field in fourier space
        self._f_padded: np.ndarray = None  # fourier transform of padded field
        self._f_reversed: np.ndarray = None  # time-reversed field in fourier space
        self._fft_axes = fft_axes
        self._fft_shape = fft_shape
        self._fft_workers = -1
        self._logger = logger if logger is not None else logging.getLogger(self.__class__.__name__)
        self._field_name = field_name

    def invalidate_f(self, also_c: bool = False):
        if also_c:
            self._c = None
        self._f = None
        self._f_padded = None
        self._f_reversed = None

    def __getitem__(self, s):
        return CachingFFT_Sliced(self, s)

    def __imul__(self, other):
        self.c *= other.c if hasattr(other, 'c') else other
        self.invalidate_f()
        return self

    def __itruediv__(self, other):
        self.c /= other.c if hasattr(other, 'c') else other
        self.invalidate_f()
        return self

    def __neg__(self) -> np.ndarray:
        # pylint: disable=invalid-unary-operand-type
        return -self.c

    def __sub__(self, other) -> np.ndarray:
        return self.c - other

    def sum(self, *args, **kwargs):
        return self.c.sum(*args, **kwargs)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.c.shape

    @property
    def has_c(self) -> bool:
        """Check if the field in coordinate space has already been computed"""
        return self._c is not None

    @property
    def has_f(self) -> bool:
        """Check if the field in fourier space has already been computed"""
        return self._f is not None

    @property
    def c(self) -> np.ndarray:
        """Getter for field in coordinate space"""
        if not self.has_c:
            self._logger.debug(f'Computing {self._field_name}(x) = FFT^-1[ {self._field_name}(f) ]', )
            assert self.has_f
            self._c = irfftn(self._f, axes=self._fft_axes, s=self._fft_shape, workers=self._fft_workers)
        return self._c

    @c.setter
    def c(self, c: np.ndarray):
        """Setter for field in coordinate space"""
        self._logger.debug(f'{"Setting" if c is not None else "Clearing"} {self._field_name}(x)')
        self._c = c
        self.invalidate_f()

    @property
    def f(self) -> np.ndarray:
        """Getter for field in fourier space"""
        if not self.has_f:
            self._logger.debug(f'Computing {self._field_name}(f) = FFT[ {self._field_name}(x) ]')
            assert self.has_c
            self._f = rfftn(self._c, axes=self._fft_axes, s=self._fft_shape, workers=self._fft_workers)
        return self._f

    def f_padded(self, pad_mode: Dict = None, pad_width: Tuple[Tuple[int, int], ...] = None) -> np.ndarray:
        """Getter for padded field in fourier space"""
        if self._f_padded is None:
            self._logger.debug(f'Computing {self._field_name}_padded(f) = FFT[ {self._field_name}_padded(x) ]')
            assert self.has_c

            c = np.pad(self._c, pad_width, **pad_mode)
            # TODO: we should actually make sure that the padding does not change between calls
            self._f_padded = rfftn(c, axes=self._fft_axes, s=self._fft_shape, workers=self._fft_workers)
        return self._f_padded

    @f.setter
    def f(self, f: np.ndarray):
        """Setter for field in fourier space"""
        self._logger.debug(f'{"Setting" if f is not None else "Clearing"} {self._field_name}(f)')
        self.invalidate_f(also_c=True)
        self._f = f

    @property
    def f_reversed(self) -> np.ndarray:
        """Getter for time-reversed field in fourier space, intentionally no setter for now"""
        if self._f_reversed is None:
            self._logger.debug(f'Computing {self._field_name}_rev(f) = FFT[ {self._field_name}(-x) ]')
            assert self.has_c
            c_reversed = np.flip(self._c, axis=self._fft_axes)
            self._f_reversed = rfftn(c_reversed, axes=self._fft_axes, s=self._fft_shape, workers=self._fft_workers)
        return self._f_reversed


class CachingFFT_Sliced(CachingFFT):
    """
    Proxy class for CachingFFT that provides access to array slices of the original object
    and keeps the caching logic intact
    """
    def __init__(self, parent: CachingFFT, s: slice):
        super().__init__(
            field_name=parent._field_name + '_sliced', c=parent._c[s],
            fft_axes=parent._fft_axes, fft_shape=parent._fft_shape,
            logger=parent._logger)
        self._parent = parent

    def invalidate_f(self, also_c: bool = False):
        assert not also_c  # clearing c in sliced objects is not supported
        super().invalidate_f(also_c)
        self._parent.invalidate_f(also_c)


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
        axes_W_normalization: Optional[Union[int, Tuple[int, ...]]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:

        w, h = super()._initialize_matrices(V, atom_shape, n_atoms, W, axes_W_normalization)

        self._V = CachingFFT(
            field_name='V', c=V,
            fft_axes=self._shift_dimensions, fft_shape=self.fft_params['reconstruct']['fft_shape'],
            logger=self._logger)

        H = CachingFFT(
            field_name='H', c=h,
            fft_axes=self._shift_dimensions, fft_shape=self.fft_params['grad_H']['fft_shape'],
            logger=self._logger)

        if W is None:
            W = CachingFFT(
                field_name='W', c=w,
                fft_axes=self._shift_dimensions, fft_shape=self.fft_params['grad_W']['fft_shape'],
                logger=self._logger)

        return W, H

    @staticmethod
    def to_ndarray(arr: CachingFFT) -> np.ndarray:
        return arr.c

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
            result = CachingFFT(field_name=n, fft_axes=fft_axes, fft_shape=fft_shape)
            result.f = contraction(a[arr1_slice], arr2_fft[arr2_slice])
            result.c = result.c[slices]
            ret += (result, )
        return ret

    def reconstruction_gradient_W(
        self,
        V: np.ndarray,
        W: CachingFFT,
        H: CachingFFT,
        s: slice = sliceNone
    ) -> Tuple[CachingFFT, CachingFFT]:

        H_, V_ = H[s], self._V[s]
        R = self.reconstruct(W, H_)
        assert R.c.shape == V_.c.shape
        neg, pos = self._fft_convolve(('neg_W', 'pos_W'), (V_, R), H_, **self.fft_params['grad_W'])
        return neg.c, pos.c

    def reconstruction_gradient_H(
        self,
        V: np.ndarray,
        W: CachingFFT,
        H: CachingFFT,
        s: slice = sliceNone
    ) -> Tuple[CachingFFT, CachingFFT]:

        H_, V_ = H[s], self._V[s]
        R = self.reconstruct(W, H_)
        assert R.c.shape == V_.c.shape
        neg, pos = self._fft_convolve(('neg_H', 'pos_H'), (V_, R), W, **self.fft_params['grad_H'])
        return neg.c, pos.c

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
