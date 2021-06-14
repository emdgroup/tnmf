"""
A module that provides some specializations and utilities for all PyTorch based backends.
"""

from typing import Tuple, Optional

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
    def _initialize_matrices(
        self,
        V: np.ndarray,
        atom_shape: Tuple[int, ...],
        n_atoms: int,
        W: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:

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
