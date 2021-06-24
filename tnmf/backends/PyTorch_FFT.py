"""
A module that provides a PyTorch based backend for computing the gradients of the factorization model.
Shift-invariance is implemented via efficient convolution operations in Fourier space.
"""

import torch
from torch import Tensor
from torch.nn.functional import pad
from torch.fft import rfftn, irfftn

from ._PyTorchBackend import PyTorchBackend


class PyTorch_FFT_Backend(PyTorchBackend):
    r"""
    Reconstruction is performed via an efficient convolution in Fourier space.
    """

    def reconstruct(self, W: Tensor, H: Tensor) -> Tensor:
        """
        Compute sum_m    H[n, m, _, ... ] * W[_ , m, c, ...]   --> R[n, c, ...]
        """

        n_shift_dimensions = W.ndim - 2

        if self._padding is None:
            H_padded = H
        else:
            H_padded = pad(H, **self._padding)

        shape = H_padded.shape[-n_shift_dimensions:]
        fft_shape = shape

        # if dim is not specified, the last len(s) dimensions are transformed, which is exactly what we need
        W_fft = rfftn(W, s=fft_shape)
        H_fft = rfftn(H_padded, s=fft_shape)

        # insert singleton dimensions to ensure broadcasting works as required
        W_fft = torch.unsqueeze(W_fft, 0)
        H_fft = torch.unsqueeze(H_fft, 2)

        R_fft = W_fft * H_fft

        R_fft = R_fft.sum(dim=1, keepdim=False)

        fslice = (slice(None), ) * 2
        fslice += tuple(slice(min(H_padded.shape[a], W.shape[a]) - 1,
                              max(H_padded.shape[a], W.shape[a])) for a in range(-n_shift_dimensions, 0))

        R = irfftn(R_fft, s=shape)

        return R[fslice]
