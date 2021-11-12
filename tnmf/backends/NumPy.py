"""
A module that provides a NumPy based backend for computing the gradients of the factorization model.
Shift-invariance is implemented via explicit convolution operations in the coordinate space.
"""
from itertools import product
from typing import Tuple, Optional, Union

import numpy as np
from numpy.lib.stride_tricks import as_strided
from opt_einsum import contract

from ._Backend import sliceNone
from ._NumPyBackend import NumPyBackend


class NumPy_Backend(NumPyBackend):
    r"""
    A plain NumPy backend for computing the gradients of the factorization model in coordinate space (no FFT, no PyTorch).

    Convolutions are computed efficiently as contractions of properly strided arrays.
    """
    def __init__(
        self,
        reconstruction_mode: str = 'valid',
    ):
        if reconstruction_mode != 'valid':
            raise NotImplementedError('This backend only supports the "valid" reconstruction mode.')
        super().__init__(reconstruction_mode=reconstruction_mode)
        self._shift_axes = None
        self._cache = {}

    def _initialize_matrices(
        self,
        V: np.ndarray,
        atom_shape: Tuple[int, ...],
        n_atoms: int,
        W: Optional[np.ndarray] = None,
        axes_W_normalization: Optional[Union[int, Tuple[int, ...]]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:

        W, H = super()._initialize_matrices(V, atom_shape, n_atoms, W, axes_W_normalization)

        n_shift_axes = len(self._sample_shape)
        self._shift_axes = tuple(range(-n_shift_axes, 0))
        shift_dim_idx = tuple(range(n_shift_axes))

        self._cache = {
            # zero-padding of the signal matrix for full-size correlation
            'pad_width': ((0, 0), (0, 0)) + tuple((a - 1, a - 1) for a in atom_shape)
        }

        self._cache.update({
            'V_padded': np.pad(V, pad_width=self._cache['pad_width']),
            # dimension labels of the data and reconstruction matrices
            'V_labels': ['n', 'c'] + ['d' + str(i) for i in shift_dim_idx],
            'W_labels': ['m', 'c'] + ['a' + str(i) for i in shift_dim_idx],
            'H_labels': ['n', 'm'] + ['d' + str(i) for i in shift_dim_idx],
            # labels for striding in gradient_H computation
            'X_strided_W_labels': ['n', 'c'] + [s + str(i) for s, i in product(['d', 'a'], shift_dim_idx)],
            # labels for striding in gradient_W computation
            'H_strided_V_labels': ['n', 'm'] + [s + str(i) for s, i in product(['a', 'd'], shift_dim_idx)],
            # dimension info for striding in reconstruction computation
            'H_strided_W_shape_suffix': V.shape[2:] + atom_shape,  # will have to be prefixed with H.shape[:2]
            'H_strided_W_labels': ['n', 'm'] + [s + str(i) for s, i in product(['d', 'a'], shift_dim_idx)],
        })

        return W, H

    def reconstruction_gradient_W(
        self,
        V: np.ndarray,
        W: np.ndarray,
        H: np.ndarray,
        s: slice = sliceNone
    ) -> Tuple[np.ndarray, np.ndarray]:
        # do NOT put these into self._cache
        H_strided_V_strides = H[s].strides + H[s].strides[2:]
        H_strided_V_shape = H[s].shape[:2] + self.atom_shape + V[s].shape[2:]
        H_strided = as_strided(H[s], H_strided_V_shape, H_strided_V_strides, writeable=False)
        R = self.reconstruct(W, H[s])

        neg = np.flip(contract(
            H_strided, self._cache['H_strided_V_labels'],
            V[s], self._cache['V_labels'],
            self._cache['W_labels'], optimize='optimal'), axis=self._shift_axes)

        pos = np.flip(contract(
            H_strided, self._cache['H_strided_V_labels'],
            R, self._cache['V_labels'],
            self._cache['W_labels'], optimize='optimal'), axis=self._shift_axes)
        return neg, pos

    def reconstruction_gradient_H(
        self,
        V: np.ndarray,
        W: np.ndarray,
        H: np.ndarray,
        s: slice = sliceNone
    ) -> Tuple[np.ndarray, np.ndarray]:

        V_padded = self._cache['V_padded'][s]
        # do NOT put these strides into self._cache (layout can change e.g. because of varying minibatch size)
        V_strided_W_strides = V_padded.strides + V_padded.strides[2:]
        V_strided_W_shape = V_padded.shape[:2] + H[s].shape[2:] + self.atom_shape
        V_strided = as_strided(V_padded, V_strided_W_shape, V_strided_W_strides, writeable=False)
        neg = contract(
            W, self._cache['W_labels'],
            V_strided, self._cache['X_strided_W_labels'],
            self._cache['H_labels'], optimize='optimal')

        R_padded = np.pad(self.reconstruct(W, H[s]), pad_width=self._cache['pad_width'])
        # do NOT put these strides into self._cache (layout can change e.g. because of varying minibatch size)
        R_strided_W_strides = R_padded.strides + R_padded.strides[2:]
        R_strided_W_shape = R_padded.shape[:2] + H[s].shape[2:] + self.atom_shape
        R_strided = as_strided(R_padded, R_strided_W_shape, R_strided_W_strides, writeable=False)
        pos = contract(
            W, self._cache['W_labels'],
            R_strided, self._cache['X_strided_W_labels'],
            self._cache['H_labels'], optimize='optimal')
        return neg, pos

    def reconstruct(self, W: np.ndarray, H: np.ndarray) -> np.ndarray:
        # do NOT put these strides into self._cache as H layout can change
        H_strided_W_strides = H.strides + H.strides[2:]
        # do NOT put into self._cache, this will break partial_reconstruct
        H_strided_W_shape = H.shape[:2] + self._cache['H_strided_W_shape_suffix']
        H_strided = as_strided(H, H_strided_W_shape, H_strided_W_strides, writeable=False)
        R = contract(
            H_strided, self._cache['H_strided_W_labels'],
            np.flip(W, self._shift_axes), self._cache['W_labels'],
            self._cache['V_labels'], optimize='optimal')
        return R
