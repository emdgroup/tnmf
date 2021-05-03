# pylint: disable=abstract-method

from typing import Tuple, Optional

import numpy as np
import torch
from torch import Tensor

from ._Backend import Backend

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


class PyTorchBackend(Backend):

    def _initialize_matrices(
        self,
        V: np.ndarray,
        atom_shape: Tuple[int, ...],
        n_atoms: int,
        W: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:

        dtype = numpy_to_torch_dtype_dict[V.dtype]
        H = (1 - torch.rand((self.n_samples, n_atoms, *self._transform_shape), dtype=dtype))

        if W is None:
            W = (1 - torch.rand((n_atoms, self.n_channels, *atom_shape), dtype=dtype))

        return W, H
