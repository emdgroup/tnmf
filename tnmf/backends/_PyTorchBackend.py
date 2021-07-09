"""
A module that provides some specializations and utilities for all PyTorch based backends.
"""

# TODO: it should be possible to reformulate the gradients using
#       https://pytorch.org/docs/stable/autograd.html#torch.autograd.functional.jacobian
# TODO: merge gradient functions into one
# TODO: add device option

from typing import Tuple, Optional, Union
from itertools import chain

import numpy as np
import torch
from torch import Tensor
from torch.nn.functional import conv1d

from ._Backend import Backend


# pylint: disable=abstract-method
class PyTorchBackend(Backend):
    r"""
    The parent class for all PyTorch based backends.

    They provide the functionality to evaluate the gradients of the factorization model via automatic differentiation
    using :mod:`torch.autograd`.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._padding = None

    def _initialize_matrices(
        self,
        V: np.ndarray,
        atom_shape: Tuple[int, ...],
        n_atoms: int,
        W: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:

        if self._reconstruction_mode == 'valid':
            self._padding = None
        elif self._reconstruction_mode == 'full':
            pad_shape = tuple(chain.from_iterable((a - 1, a - 1) for a in reversed(atom_shape)))
            self._padding = dict(pad=pad_shape, mode='constant', value=0)
        elif self._reconstruction_mode in ('circular', 'reflect'):
            pad_shape = tuple(chain.from_iterable((a - 1, 0) for a in reversed(atom_shape)))
            self._padding = dict(pad=pad_shape, mode=self._reconstruction_mode)
        else:
            raise ValueError(f'Unsupported reconstruction mode "{self._reconstruction_mode}".'
                             f'Please choose "valid", "full", "circular", or "reflect".')

        w, h = super()._initialize_matrices(V, atom_shape, n_atoms, W)

        H = torch.from_numpy(h)
        if W is None:
            W = torch.from_numpy(w)

        return W, H

    @staticmethod
    def to_ndarray(arr: Tensor) -> np.ndarray:
        return arr.detach().numpy()

    @staticmethod
    def convolve_multi_1d(arr: Tensor, kernels: Tuple[np.ndarray, ...], axes: Tuple[int, ...]) -> Tensor:
        assert len(kernels) == len(axes)

        convolved = arr.detach().clone()
        for a, kernel in zip(axes, kernels):
            pad = (len(kernel) - 1) // 2
            kernel_view = torch.as_tensor(kernel).view(1, 1, -1)
            # axis to be convolve across has to be the last one
            convolved = torch.movedim(convolved, a, -1)
            convolved_shape = convolved.size()
            # all other non-singleton axes have to be aggregated into the batch_size axis (0th axis)
            convolved = torch.reshape(convolved, (-1, 1, convolved.size()[-1]))
            convolved = conv1d(convolved, kernel_view, padding=pad)
            # restore original shape
            convolved = torch.reshape(convolved, convolved_shape)
            # restore original axis order
            convolved = torch.movedim(convolved, -1, a)

        return convolved

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

    def reconstruction_energy(self, V: Tensor, W: Tensor, H: Tensor) -> float:
        V = torch.as_tensor(V)
        R = self.reconstruct(W, H)
        energy = 0.5 * torch.sum(torch.square(V - R))
        return float(energy)
