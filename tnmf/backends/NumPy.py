from .Backend import Backend
import numpy as np
from numpy.lib.stride_tricks import as_strided
from opt_einsum import contract
from itertools import product
from typing import Tuple, Optional


class NumPy_Backend(Backend):

    def initialize_matrices(
            self,
            V: np.ndarray,
            atom_shape: Tuple[int, ...],
            n_atoms: int,
            mode_R: str,
            W: Optional[np.array] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:

        W, H = super().initialize_matrices(V, atom_shape, n_atoms, mode_R, W)

        n_shift_dimensions = len(self._sample_shape)

        self._cache = {
            # zero-padding of the signal matrix for full-size correlation
            'pad_width': ((0, 0), (0, 0), *n_shift_dimensions * (tuple(np.array(atom_shape) - 1), )),
        }

        self._cache.update({
            'V_padded': np.pad(V, pad_width=self._cache['pad_width']),
        })

        # TODO: fix indices and clean from here
        self._cache.update({
            # dimension labels of the data and reconstruction matrices
            'V_labels': ['n', 'c'] + ['d' + str(i) for i in range(n_shift_dimensions)],
            'W_labels': ['m', 'c'] + ['a' + str(i) for i in range(n_shift_dimensions)],
            'H_labels': ['n', 'm'] + ['d' + str(i) for i in range(n_shift_dimensions)],
            # dimension info for striding in gradient_H computation
            'X_strided_W_shape': atom_shape + H.shape[2:] + V.shape[2:],
            'X_strided_W_strides': self._cache['V_padded'].strides[-n_shift_dimensions:] + self._cache['V_padded'].strides,
            'X_strided_W_labels': ['n', 'c'] + [s + str(i) for s, i in product(['d', 'a'], range(n_shift_dimensions))],
            # dimension info for striding in gradient_W computation
            'H_strided_V_shape': V.shape[-n_shift_dimensions:] + atom_shape + H.shape[2:],
            'H_strided_V_strides': H.strides[-n_shift_dimensions:] + H.strides,
            'H_strided_V_labels': ['n', 'm'] + [s + str(i) for s, i in product(['a', 'd'], range(n_shift_dimensions))],
            # dimension info for striding in reconstruction computation
            'H_strided_W_shape': atom_shape + V.shape[:-n_shift_dimensions] + H.shape[2:],
            'H_strided_W_strides': H.strides[-n_shift_dimensions:] + H.strides,
            'H_strided_W_labels': ['n', 'm'] + [s + str(i) for s, i in product(['d', 'a'], range(n_shift_dimensions))],
        })

        return W, H

    def reconstruction_gradient_W(self, V: np.array, W: np.array, H: np.array) -> Tuple[np.array, np.array]:
        H_strided = as_strided(H, self._cache['H_strided_V_shape'], self._cache['H_strided_V_strides'], writeable=False)
        R = self.reconstruct(W, H)
        numer = np.flip(contract(H_strided, self._cache['H_strided_V_labels'], V, self._cache['V_labels'], self._cache['W_labels'], optimize='optimal'), axis=self._sample_shape)
        denum = np.flip(contract(H_strided, self._cache['H_strided_V_labels'], R, self._cache['V_labels'], self._cache['W_labels'], optimize='optimal'), axis=self._sample_shape)
        return numer, denum

    def reconstruction_gradient_H(self, V: np.array, W: np.array, H: np.array) -> Tuple[np.array, np.array]:
        V_padded = self._cache['V_padded']
        R = self.reconstruct(W, H)
        R_padded = np.pad(self.R, pad_width=self._cache['pad_width'])
        V_strided = as_strided(V_padded, self._cache['X_strided_W_shape'], self._cache['X_strided_W_strides'], writeable=False)
        R_strided = as_strided(R_padded, self._cache['X_strided_W_shape'], self._cache['X_strided_W_strides'], writeable=False)
        numer = contract(W, self._cache['W_labels'], V_strided, self._cache['X_strided_W_labels'], self._cache['H_labels'], optimize='optimal')
        denum = contract(W, self._cache['W_labels'], R_strided, self._cache['X_strided_W_labels'], self._cache['H_labels'], optimize='optimal')
        return numer, denum

    def reconstruct(self, W: np.array, H: np.array) -> np.array:
        H_strided = as_strided(H, self._cache['H_strided_W_shape'], self._cache['H_strided_W_strides'], writeable=False)
        R = contract(H_strided, self._cache['H_strided_W_labels'], np.flip(W, self._sample_shape), self._cache['W_labels'], self._cache['V_labels'], optimize='optimal')
        return R
