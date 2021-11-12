"""
A module that provides some specializations and utilities for all NumPy based backends
that are using FFT for performing convolutions.
"""
from typing import Tuple, Optional, Union

import numpy as np
from scipy.fft import next_fast_len
from opt_einsum import contract_expression

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
        axes_W_normalization: Optional[Union[int, Tuple[int, ...]]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:

        # this sets pad_mode and pad_width properly
        w, h = super()._initialize_matrices(V, atom_shape, n_atoms, W, axes_W_normalization)

        # shorthand for unpadded and unsliced axes
        unpadded = ((0, 0), ) * (V.ndim - len(self._shift_dimensions))
        unsliced = (slice(None), ) * (V.ndim - len(self._shift_dimensions))
        # shape for fft
        fft_shape = tuple(next_fast_len(s) for s in np.array(self._sample_shape) + np.array(self._transform_shape) - 1)
        # shorthands for shape of FFT fields
        H_f_shape = (self.n_samples, n_atoms) + fft_shape
        W_f_shape = (n_atoms, self.n_channels) + fft_shape
        V_f_shape = (self.n_samples, self.n_channels) + fft_shape

        # fft details: reconstruction
        lower_idx = np.array(atom_shape) - 1
        self.fft_params['reconstruct'] = {
            'fft_axes': self._shift_dimensions,
            'pad_mode': self._pad_mode,
            'pad_width': unpadded + self._padding_left,
            'correlate': False,
            'slices': unsliced + tuple(slice(f, f + s) for f, s in zip(lower_idx, self._sample_shape)),
            'fft_shape': fft_shape,
            # sum_c V|R[n, c, ... ] * W[m , c, ...] --> dR / dH[n, m, ...]
            'contraction': contract_expression('nm...,mc...->nc...', H_f_shape, W_f_shape)
        }

        # fft details: gradient H computation
        lower_idx = np.zeros_like(self._transform_shape)
        if self._pad_mode is not None:
            lower_idx += np.asarray(self._padding_right)[:, 1]
        self.fft_params['grad_H'] = {
            'fft_axes': self._shift_dimensions,
            'pad_mode': self._pad_mode,
            'pad_width': unpadded + self._padding_right,
            'correlate': True,
            'slices': unsliced + tuple(slice(f, f + s) for f, s in zip(lower_idx, self._transform_shape)),
            'fft_shape': fft_shape,
            # sum_c V|R[n, c, ... ] * W[m , c, ...] --> dR / dH[n, m, ...]
            'contraction': contract_expression('nc...,mc...->nm...', V_f_shape, W_f_shape),
        }

        # fft details: gradient W computation
        lower_idx = np.minimum(np.array(self._sample_shape), np.array(self._transform_shape)) - 1
        self.fft_params['grad_W'] = {
            'fft_axes': self._shift_dimensions,
            'pad_mode': self._pad_mode,
            'pad_width': unpadded + self._padding_right,
            'correlate': True,
            'slices': unsliced + tuple(slice(f, f + s) for f, s in zip(lower_idx, atom_shape)),
            'fft_shape': fft_shape,
            # sum_n V|R[n, c, ... ] * H[n, m, ...]   --> dR / dW[m, c, ...]
            'contraction': contract_expression('nc...,nm...->mc...', V_f_shape, H_f_shape),
        }

        return w, h
