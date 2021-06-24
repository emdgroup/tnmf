"""
A module that provides a PyTorch based backend for computing the gradients of the factorization model.
Shift-invariance is implemented via explicit convolution operations in the coordinate space.
"""

# TODO: use torch.fft.rfftn() to generalize for more dimensions and improve performance

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
    Reconstruction is performed via an explicit convolution in coordinate space.
    """

    def reconstruct(self, W: Tensor, H: Tensor) -> Tensor:
        # TODO: support dimensions > 3
        # TODO: consider transposed convolution as alternative

        n_shift_dimensions = W.ndim - 2

        assert n_shift_dimensions <= 3
        conv_fun = _CONV_DICT[n_shift_dimensions]
        flip_dims = list(range(-n_shift_dimensions, 0))
        W_flipped = torch.flip(W, flip_dims)

        if self._padding is None:
            H_padded = H
        else:
            H_padded = pad(H, **self._padding)

        R = conv_fun(H_padded, torch.swapaxes(W_flipped, 0, 1))
        return R
