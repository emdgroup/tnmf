# TODO: generalize n_transforms from numpy_fft to all backends
# TODO: all backends need to support self._mode_R et al
# TODO: do we need self._input_padding ? If yes, all backends have to support it.
# TODO: refactor common backend logic of NumpyBackend/PyTorchBackend into function

import abc
from typing import Tuple, Optional, Dict, Union

import numpy as np


class Backend(abc.ABC):

    def __init__(
        self,
        reconstruction_mode: str = 'valid',
        input_padding: Dict = None,
    ):
        self._input_padding = input_padding if input_padding is not None else dict(mode='constant', constant_values=0)
        self._reconstruction_mode = reconstruction_mode
        self.n_samples = None
        self.n_channels = None
        self._sample_shape = None
        self._transform_shape = None
        self._n_shift_dimensions = None
        self._shift_dimensions = None

    def initialize(
        self,
        V: np.ndarray,
        atom_shape: Tuple[int, ...],
        n_atoms: int,
        W: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        self._set_dimensions(V, atom_shape)
        return self._initialize_matrices(V, atom_shape, n_atoms, W)

    @staticmethod
    @abc.abstractmethod
    def to_ndarray(arr) -> np.ndarray:
        raise NotImplementedError

    def _set_dimensions(self, V, atom_shape):
        self.n_samples = V.shape[0]
        self.n_channels = V.shape[1]
        self._sample_shape = V.shape[2:]
        self._transform_shape = self.n_transforms(self._sample_shape, atom_shape)
        self._n_shift_dimensions = len(atom_shape)
        self._shift_dimensions = tuple(range(-1, -len(atom_shape) - 1, -1))

    def n_transforms(self, sample_shape: Tuple[int, ...], atom_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        # TODO: remove or rename this function
        """Number of dictionary transforms in each dimension"""
        if self._reconstruction_mode == 'valid':
            return tuple(np.array(sample_shape) + np.array(atom_shape) - 1)

        if self._reconstruction_mode == 'full':
            return tuple(np.array(sample_shape) - np.array(atom_shape) + 1)

        if self._reconstruction_mode in ('same', 'circular'):
            return tuple(np.array(sample_shape))

        raise ValueError

    @staticmethod
    def normalize(arr: np.ndarray, axis: Optional[Union[int, Tuple[int, ...]]] = None):
        arr /= (arr.sum(axis=axis, keepdims=True))

    @staticmethod
    def convolve_multi_1d(arr: np.ndarray, kernels: Tuple[np.ndarray, ...], axes: Tuple[int, ...]) -> np.ndarray:
        raise NotImplementedError

    def _initialize_matrices(
        self,
        V: np.ndarray,
        atom_shape: Tuple[int, ...],
        n_atoms: int,
        W: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:

        H = np.asarray(1 - np.random.rand(self.n_samples, n_atoms, *self._transform_shape), dtype=V.dtype)

        if W is None:
            W = np.asarray(1 - np.random.rand(n_atoms, self.n_channels, *atom_shape), dtype=V.dtype)

        return W, H

    @abc.abstractmethod
    def reconstruction_gradient_W(self, V: np.ndarray, W: np.ndarray, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    @abc.abstractmethod
    def reconstruction_gradient_H(self, V: np.ndarray, W: np.ndarray, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    @abc.abstractmethod
    def reconstruct(self, W: np.ndarray, H: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def partial_reconstruct(self, W: np.ndarray, H: np.ndarray, i_atom: int) -> np.ndarray:
        return self.reconstruct(W[i_atom:i_atom+1], H[:, i_atom:i_atom+1])

    def reconstruction_energy(self, V: np.ndarray, W: np.ndarray, H: np.ndarray) -> float:
        R = self.to_ndarray(self.reconstruct(W, H))
        assert R.shape == V.shape
        return 0.5 * np.sum(np.square(V - R))
