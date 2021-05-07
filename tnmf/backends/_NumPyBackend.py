# pylint: disable=abstract-method

import numpy as np

from ._Backend import Backend


class NumPyBackend(Backend):

    @staticmethod
    def to_ndarray(arr: np.ndarray) -> np.ndarray:
        return arr
