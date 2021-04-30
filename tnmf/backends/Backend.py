# TODO: generalize n_transforms from numpy_fft to all backends
# TODO: all backends need to support self._mode_R et al
# TODO: do we need self._input_padding ? If yes, all backends have to support it.
# TODO: fix numpy to torch dtype
# TODO: refactor common backend logic of NumpyBackend/PyTorchBackend into function

import abc
import torch
import numpy as np
from typing import Tuple, Optional
from torch import Tensor

# see https://github.com/pytorch/pytorch/issues/40568#issuecomment-649961327
numpy_to_torch_dtype_dict = {
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.dtype('float64'): torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128
}


class Backend(metaclass=abc.ABCMeta):

    def __init__(
        self,
        reconstruction_mode: str = 'valid',
        input_padding=dict(mode='constant', constant_values=0),
    ):
        self._input_padding = input_padding
        self._mode_R = reconstruction_mode
        self._mode_H = {'full': 'valid', 'valid': 'full', 'same': 'same', }[reconstruction_mode]

    def _get_dimensions(self, V, atom_shape):
        self.n_samples = V.shape[0]
        self.n_channels = V.shape[1]
        self._sample_shape = V.shape[2:]
        self._transform_shape = self.n_transforms(self._sample_shape, atom_shape)
        self._n_shift_dimensions = len(atom_shape)
        self._shift_dimensions = tuple(range(-1, -len(atom_shape)-1, -1))

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


class NumpyBackend(Backend):

    def initialize_matrices(
            self,
            V: np.ndarray,
            atom_shape: Tuple[int, ...],
            n_atoms: int,
            W: Optional[np.array] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:

        H = np.asarray(1 - np.random.rand(self.n_samples, n_atoms, *self._transform_shape), dtype=V.dtype)

        if W is None:
            W = np.asarray(1 - np.random.rand(n_atoms, self.n_channels, *atom_shape), dtype=V.dtype)

        return W, H


class PyTorchBackend(Backend):

    def initialize_matrices(
            self,
            V: np.ndarray,
            atom_shape: Tuple[int, ...],
            n_atoms: int,
            W: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:

        self.dtype = numpy_to_torch_dtype_dict[V.dtype]
        H = (1 - torch.rand((self.n_samples, n_atoms, *self._transform_shape), dtype=self.dtype))

        if W is None:
            W = (1 - torch.rand((n_atoms, self.n_channels, *atom_shape), dtype=self.dtype))

        return W, H
