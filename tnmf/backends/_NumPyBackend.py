"""
A module that provides some specializations and utilities for all NumPy based backends.
"""

from typing import Tuple, Optional

import numpy as np
from scipy.ndimage import convolve1d

from ._Backend import Backend


# pylint: disable=abstract-method
class NumPyBackend(Backend):
    r"""
    The parent class for all NumPy based backends.

    They provide the functionality to evaluate the analytic gradients of the factorization model.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._pad_mode = None
        self._padding_left = None
        self._padding_right = None

    def _initialize_matrices(
        self,
        V: np.ndarray,
        atom_shape: Tuple[int, ...],
        n_atoms: int,
        W: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:

        self._padding_left = tuple((s - 1, 0) for s in self.atom_shape)
        self._padding_right = tuple((0, s - 1) for s in self.atom_shape)

        if self._reconstruction_mode == 'valid':
            self._pad_mode = None
        elif self._reconstruction_mode == 'full':
            self._pad_mode = dict(mode='constant', constant_values=0.)
        elif self._reconstruction_mode == 'circular':
            self._pad_mode = dict(mode='wrap')
        elif self._reconstruction_mode == 'reflect':
            self._pad_mode = dict(mode='reflect', reflect_type='even')
        else:
            raise ValueError(f'Unsupported reconstruction mode "{self._reconstruction_mode}".'
                             f'Please choose "valid", "full", "circular", or "reflect".')

        return super()._initialize_matrices(V, atom_shape, n_atoms, W)

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
