"""
Author: Mathias Winkel
"""

import logging
import numpy as np
from scipy.fft import rfftn, irfftn
from typing import Optional, Tuple, List


class CachingFFT(object):
    """
    Wrapper class for conveniently caching and switching back and forth
    between fields in coordinate space and fourier space
    """

    def __init__(self, field_name: str,
                 fft_axes: Optional[Tuple[int]] = None,
                 fft_shape: Optional[List[int]] = None, logger: Optional[logging.Logger] = None):
        self._c = None  # field in coordinate space
        self._f = None  # field in fourier space
        self._f_reversed = None  # time-reversed field in fourier space
        self._fft_axes = fft_axes
        self._fft_shape = fft_shape
        self._fft_workers = -1
        self._logger = logger if logger is not None else logging.getLogger(self.__class__.__name__)
        self._field_name = field_name

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
