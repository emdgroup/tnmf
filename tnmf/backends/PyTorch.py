# TODO: fix numpy to torch dtype
# TODO: it should be possible to reformulate the gradients using
#       https://pytorch.org/docs/stable/autograd.html#torch.autograd.functional.jacobian
# TODO: merge gradient functions into one
# TODO: add device option

from .Backend import Backend
from torch import Tensor
import torch
import numpy as np
from typing import Tuple, Optional

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


class PyTorch_Backend(Backend):

    def initialize_matrices(
            self,
            V: np.ndarray,
            atom_shape: Tuple[int, ...],
            n_atoms: int,
            W: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        self._sample_shape = V.shape[2:]
        n_samples = V.shape[0]
        n_channels = V.shape[1]

        transform_shape = self.n_transforms(self._sample_shape, atom_shape)

        self.dtype = numpy_to_torch_dtype_dict[V.dtype]
        H = (1 - torch.rand((n_samples, n_atoms, *transform_shape), dtype=self.dtype)).requires_grad_()

        if W is None:
            W = (1 - torch.rand((n_atoms, n_channels, *atom_shape), dtype=self.dtype)).requires_grad_()

        return W, H

    def normalize(self, arr: Tensor, axes: Tuple[int]) -> Tensor:
        return arr / (arr.sum(dim=axes, keepdim=True))

    def reconstruction_gradient_W(self, V: np.ndarray, W: Tensor, H: Tensor) -> Tuple[Tensor, Tensor]:
        neg_energy, pos_energy = self._energy_terms(V, W, H)
        neg = torch.autograd.grad(neg_energy, W, retain_graph=True)[0]
        pos = torch.autograd.grad(pos_energy, W)[0]
        return neg, pos

    def reconstruction_gradient_H(self, V: np.ndarray, W: Tensor, H: Tensor) -> Tuple[Tensor, Tensor]:
        neg_energy, pos_energy = self._energy_terms(V, W, H)
        neg = torch.autograd.grad(neg_energy, H, retain_graph=True)[0]
        pos = torch.autograd.grad(pos_energy, H)[0]
        return neg, pos

    def _energy_terms(self, V: np.ndarray, W: Tensor, H: Tensor) -> Tuple[Tensor, Tensor]:
        V = torch.as_tensor(V)
        R = self._reconstruct_torch(W, H)
        neg = (R * V).sum()
        pos = 0.5 * (V.square().sum() + R.square().sum())
        return neg, pos

    def _reconstruct_torch(self, W: Tensor, H: Tensor) -> Tensor:
        n_samples = H.shape[0]
        n_atoms = W.shape[0]
        n_channels = W.shape[1]

        # TODO: support multiple dimensions
        # TODO: remove for loops
        assert len(self._sample_shape) == 2

        R = torch.zeros((n_samples, n_channels, *self._sample_shape), dtype=self.dtype)

        for i_sample in range(n_samples):
            for i_channel in range(n_channels):
                for i_atom in range(n_atoms):
                    w = W[i_atom, i_channel]
                    w = torch.flip(w, (-2, -1))  # torch.nn.functional.conv2d() is actually a correlation

                    h = H[i_sample, i_atom]
                    R[i_sample] += torch.nn.functional.conv2d(h.view((1, 1, *h.shape)),
                                                              w.view((1, 1, *w.shape)))[0, 0, :, :]
            return R

    def reconstruct(self, W: Tensor, H: Tensor) -> np.ndarray:
        R = self._reconstruct_torch(W, H)
        return R.detach().numpy()

    def reconstruction_energy(self, V: Tensor, W: Tensor, H: Tensor) -> float:
        V = torch.as_tensor(V)
        R = self._reconstruct_torch(W, H)
        energy = 0.5 * torch.sum(torch.square(V - R))
        return float(energy)
