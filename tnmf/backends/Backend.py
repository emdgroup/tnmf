# TODO: generalize n_transforms from numpy_fft to all backends

import abc
import numpy as np
from typing import Tuple, Optional


class Backend(metaclass=abc.ABCMeta):

    def initialize_matrices(
            self,
            V: np.ndarray,
            atom_shape: Tuple[int, ...],
            n_atoms: int,
            transform_shape: Tuple[int, ...],
            W: Optional[np.array] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        self._sample_shape = V.shape[2:]
        n_samples = V.shape[0]
        n_channels = V.shape[1]

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