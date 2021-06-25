"""
A module that provides some specializations and utilities for all NumPy based backends.
"""

from typing import Tuple

import numpy as np
from scipy.ndimage import convolve1d

from ._Backend import Backend


# pylint: disable=abstract-method
class NumPyBackend(Backend):
    r"""
    The parent class for all NumPy based backends.

    They provide the functionality to evaluate the analytic gradients of the factorization model.
    """
    def __init__(self, reconstruction_mode: str = 'valid'):

        super().__init__(reconstruction_mode=reconstruction_mode)

        try:
            self._mode_H, self._input_padding = {
                'full': ('valid', dict(mode='constant', constant_values=0)),
                'valid': ('full', dict(mode='constant', constant_values=0)),
                'same': ('same', dict(mode='constant', constant_values=0)),
                'circular': ('same', dict(mode='wrap')),
                'reflect': ('same', dict(mode='reflect', reflect_type='even')),
                }[reconstruction_mode]
        except KeyError as e:
            raise NotImplementedError from e

    @staticmethod
    def to_ndarray(arr: np.ndarray) -> np.ndarray:
        return arr

    @staticmethod
    def convolve_multi_1d(arr: np.ndarray, kernels: Tuple[np.ndarray, ...], axes: Tuple[int, ...]) -> np.ndarray:
        assert len(kernels) == len(axes)

        convolved = arr
        for a, kernel in zip(axes, kernels):
            convolved = convolve1d(convolved, kernel, axis=a, mode='constant', cval=0.0)

        return convolved
