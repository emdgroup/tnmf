"""
A module that provides some specializations and utilities for all NumPy based backends
that are using FFT for performing convolutions.
"""
from typing import Tuple, Optional

import numpy as np

from ._NumPyBackend import NumPyBackend


class NumPyFFTBackend(NumPyBackend):
    r"""
    The parent class for all NumPy based backends that are using FFT for performing convolutions.

    They provide the functionality to evaluate the analytic gradients of the factorization model.
    """
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fft_params = {}

    def _initialize_matrices(
        self,
        V: np.ndarray,
        atom_shape: Tuple[int, ...],
        n_atoms: int,
        W: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:

        # this sets pad_mode and pad_width properly
        w, h = super()._initialize_matrices(V, atom_shape, n_atoms, W)

        # fft details: reconstruction
        lower_idx = np.array(atom_shape) - 1
        self.fft_params['reconstruct'] = {
            'fft_axes': self._shift_dimensions,
            'pad_mode': self._pad_mode,
            'pad_width': self._padding_left,
            'correlate': False,
            'slices': tuple(slice(f, f + s) for f, s in zip(lower_idx, self._sample_shape)),
        }

        # fft details: gradient H computation
        lower_idx = np.zeros_like(self._transform_shape)
        if self._pad_mode is not None:
            lower_idx += np.asarray(self._padding_right)[:, 1]
        self.fft_params['grad_H'] = {
            'fft_axes': self._shift_dimensions,
            'pad_mode': self._pad_mode,
            'pad_width': self._padding_right,
            'correlate': True,
            'slices': tuple(slice(f, f + s) for f, s in zip(lower_idx, self._transform_shape)),
        }

        # fft details: gradient W computation
        lower_idx = np.minimum(np.array(self._sample_shape), np.array(self._transform_shape)) - 1
        self.fft_params['grad_W'] = {
            'fft_axes': self._shift_dimensions,
            'pad_mode': self._pad_mode,
            'pad_width': self._padding_right,
            'correlate': True,
            'slices': tuple(slice(f, f + s) for f, s in zip(lower_idx, atom_shape)),
        }

        return w, h
