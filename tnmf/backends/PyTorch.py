# TODO: it should be possible to reformulate the gradients using
#       https://pytorch.org/docs/stable/autograd.html#torch.autograd.functional.jacobian
# TODO: merge gradient functions into one
# TODO: add device option
# TODO: use torch.fft.rfftn() to generalize for more dimensions and improve performance

from typing import Tuple, Union, Optional

import numpy as np
import torch
from torch import Tensor

from ._PyTorchBackend import PyTorchBackend


conv_dict = {
    1: torch.nn.functional.conv1d,
    2: torch.nn.functional.conv2d,
    3: torch.nn.functional.conv3d,
}


class PyTorch_Backend(PyTorchBackend):

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
        R = self._reconstruct_torch(W, H)
        neg = (R * V).sum()
        pos = 0.5 * (V.square().sum() + R.square().sum())
        return neg, pos

    def _reconstruct_torch(self, W: Tensor, H: Tensor) -> Tensor:
        # TODO: support dimensions > 3
        # TODO: remove atom for loop
        # TODO: consider transposed convolution as alternative

        n_samples = H.shape[0]
        n_atoms = W.shape[0]
        n_channels = W.shape[1]
        n_shift_dimensions = W.ndim - 2

        assert n_shift_dimensions <= 3
        conv_fun = conv_dict[n_shift_dimensions]
        flip_dims = list(range(-n_shift_dimensions, 0))
        W_flipped = torch.flip(W, flip_dims)

        R = torch.zeros((n_samples, n_channels, *self._sample_shape), dtype=W.dtype)
        for i_atom in range(n_atoms):
            R += conv_fun(H[:, i_atom, None], W_flipped[i_atom, None])

        return R

    def reconstruct(self, W: Tensor, H: Tensor) -> np.ndarray:
        R = self._reconstruct_torch(W, H)
        return R.detach().numpy()

    def reconstruction_energy(self, V: Tensor, W: Tensor, H: Tensor) -> float:
        V = torch.as_tensor(V)
        R = self._reconstruct_torch(W, H)
        energy = 0.5 * torch.sum(torch.square(V - R))
        return float(energy)
