# pylint: disable=abstract-method

from typing import Tuple, Optional

import numpy as np

from ._Backend import Backend


class NumPyBackend(Backend):

    @staticmethod
    def to_ndarray(arr: np.ndarray) -> np.ndarray:
        return arr
