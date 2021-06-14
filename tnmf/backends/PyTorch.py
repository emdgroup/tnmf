"""
A module that provides a PyTorch based backend for computing the gradients of the factorization model.
Shift-invariance is implemented via explicit convolution operations in the coordinate space.
"""

# TODO: it should be possible to reformulate the gradients using
#       https://pytorch.org/docs/stable/autograd.html#torch.autograd.functional.jacobian
# TODO: merge gradient functions into one
# TODO: add device option
# TODO: use torch.fft.rfftn() to generalize for more dimensions and improve performance

from itertools import chain
from typing import Tuple, Union, Optional

import numpy as np
import torch
from torch import Tensor
from torch.nn.functional import pad

from ._PyTorchBackend import PyTorchBackend


#: Lookup table for the convolution function with different dimensionality
_CONV_DICT = {
    1: torch.nn.functional.conv1d,
    2: torch.nn.functional.conv2d,
    3: torch.nn.functional.conv3d,
}


class PyTorch_Backend(PyTorchBackend):
    r"""
    A PyTorch based backend that uses :func:`torch.autograd.grad` for computing the gradients of the factorization model.

    Reconstruction is performed via an explicit convolution in coordinate space.
    """
    def __init__(self, reconstruction_mode: str = 'valid'):
        if reconstruction_mode not in ('valid', 'full', 'circular'):
            raise NotImplementedError
        super().__init__(reconstruction_mode=reconstruction_mode)

    @staticmethod
    def normalize(arr: Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None):
        arr.divide_(arr.sum(dim=axis, keepdim=True))

    def reconstruction_gradient_W(self, V: np.ndarray, W: Tensor, H: Tensor) -> Tuple[Tensor, Tensor]:
        W_grad = W.detach().requires_grad_()
        neg_energy, pos_energy = self._energy_terms(V, W_grad, H)
        neg = torch.autograd.grad(neg_energy, W_grad, retain_graph=True)[0]
        pos = torch.autograd.grad(pos_energy, W_grad)[0]
        return neg.detach(), pos.detach()

    def reconstruction_gradient_H(self, V: np.ndarray, W: Tensor, H: Tensor) -> Tuple[Tensor, Tensor]:
        H_grad = H.detach().requires_grad_()
        neg_energy, pos_energy = self._energy_terms(V, W, H_grad)
        neg = torch.autograd.grad(neg_energy, H_grad, retain_graph=True)[0]
        pos = torch.autograd.grad(pos_energy, H_grad)[0]
        return neg.detach(), pos.detach()

    def _energy_terms(self, V: np.ndarray, W: Tensor, H: Tensor) -> Tuple[Tensor, Tensor]:
        V = torch.as_tensor(V)
        R = self.reconstruct(W, H)
        neg = (R * V).sum()
        pos = 0.5 * (V.square().sum() + R.square().sum())
        return neg, pos

    def reconstruct(self, W: Tensor, H: Tensor) -> Tensor:
        # TODO: support dimensions > 3
        # TODO: consider transposed convolution as alternative

        n_shift_dimensions = W.ndim - 2

        assert n_shift_dimensions <= 3
        conv_fun = _CONV_DICT[n_shift_dimensions]
        flip_dims = list(range(-n_shift_dimensions, 0))
        W_flipped = torch.flip(W, flip_dims)

        pad_shape = np.array(W.shape[-n_shift_dimensions:]) - 1
        if self._reconstruction_mode == 'valid':
            H_padded = H
        elif self._reconstruction_mode == 'full':
            padding = tuple(np.repeat(pad_shape[::-1], n_shift_dimensions))
            H_padded = pad(H, padding)
        elif self._reconstruction_mode == 'circular':
            padding = tuple(chain(*((s, 0) for s in pad_shape[::-1])))
            H_padded = pad(H, padding, 'circular')
        R = conv_fun(H_padded, torch.swapaxes(W_flipped, 0, 1))
        return R

    def reconstruction_energy(self, V: Tensor, W: Tensor, H: Tensor) -> float:
        V = torch.as_tensor(V)
        R = self.reconstruct(W, H)
        energy = 0.5 * torch.sum(torch.square(V - R))
        return float(energy)
