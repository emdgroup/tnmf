# pylint: disable=abstract-method

from typing import Tuple, Optional, Union

import numpy as np
import torch
from torch import Tensor

from ._Backend import Backend


class PyTorchBackend(Backend):

    def _initialize_matrices(
        self,
        V: np.ndarray,
        atom_shape: Tuple[int, ...],
        n_atoms: int,
        W: Optional[Union[np.ndarray, Tensor]] = None,
    ) -> Tuple[Tensor, Tensor]:

        W, H = super()._initialize_matrices(V, atom_shape, n_atoms, W)

        H = torch.from_numpy(H)
        if W is not None:
            W = torch.as_tensor(W)

        return W, H

    @staticmethod
    def to_ndarray(arr: Tensor) -> np.ndarray:
        return arr.detach().numpy()
