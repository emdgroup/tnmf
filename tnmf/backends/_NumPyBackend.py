# pylint: disable=abstract-method

from typing import Tuple, Optional

import numpy as np

from ._Backend import Backend


class NumPyBackend(Backend):

    def _initialize_matrices(
        self,
        V: np.ndarray,
        atom_shape: Tuple[int, ...],
        n_atoms: int,
        W: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:

        H = np.asarray(1 - np.random.rand(self.n_samples, n_atoms, *self._transform_shape), dtype=V.dtype)

        if W is None:
            W = np.asarray(1 - np.random.rand(n_atoms, self.n_channels, *atom_shape), dtype=V.dtype)

        return W, H

    @staticmethod
    def to_ndarray(arr: np.ndarray) -> np.ndarray:
        return arr
