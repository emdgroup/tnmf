# TODO: generalize n_transforms from numpy_fft to all backends
# TODO: create numpy-specific backend class
# TODO: all backends need to support self._mode_R et al
# TODO: do we need self._input_padding ? If yes, all backends have to support it.

import abc
import numpy as np
from typing import Tuple, Optional


class Backend(metaclass=abc.ABCMeta):

    def __init__(
        self,
        reconstruction_mode: str = 'valid',
        input_padding=dict(mode='constant', constant_values=0),
    ):
        self._input_padding = input_padding
        self._mode_R = reconstruction_mode
        self._mode_H = {'full': 'valid', 'valid': 'full', 'same': 'same', }[reconstruction_mode]

    def n_transforms(self, sample_shape: Tuple[int, ...], atom_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Number of dictionary transforms in each dimension"""
        if self._mode_R == 'valid':
            return tuple(np.array(sample_shape) + np.array(atom_shape) - 1)
        elif self._mode_R == 'full':
            return tuple(np.array(sample_shape) - np.array(atom_shape) + 1)
        elif self._mode_R == 'same':
            return tuple(np.array(sample_shape))
        else:
            raise ValueError

    def initialize_matrices(
            self,
            V: np.ndarray,
            atom_shape: Tuple[int, ...],
            n_atoms: int,
            mode_R: str,
            W: Optional[np.array] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        self._sample_shape = V.shape[2:]
        n_samples = V.shape[0]
        n_channels = V.shape[1]

        transform_shape = self.n_transforms(self._sample_shape, atom_shape)

        H = np.asarray(1 - np.random.rand(n_samples, n_atoms, *transform_shape), dtype=V.dtype)

        if W is None:
            W = np.asarray(1 - np.random.rand(n_atoms, n_channels, *atom_shape), dtype=V.dtype)

        return W, H

    def normalize(self, arr: np.ndarray, axes: Tuple[int]) -> np.ndarray:
        return arr / (arr.sum(axis=axes, keepdims=True))

    @abc.abstractmethod
    def reconstruction_gradient_W(self, V: np.ndarray, W: np.ndarray, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    @abc.abstractmethod
    def reconstruction_gradient_H(self, V: np.ndarray, W: np.ndarray, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    @abc.abstractmethod
    def reconstruct(self, W: np.ndarray, H: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def reconstruction_energy(self, V: np.ndarray, W: np.ndarray, H: np.ndarray) -> float:
        R = self.reconstruct(W, H)
        assert R.shape == V.shape
        return 0.5 * np.sum(np.square(V - R))
