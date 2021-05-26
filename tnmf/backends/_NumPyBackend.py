# pylint: disable=abstract-method

from typing import Tuple

import numpy as np
from scipy.ndimage import convolve1d

from ._Backend import Backend


class NumPyBackend(Backend):

    @staticmethod
    def to_ndarray(arr: np.ndarray) -> np.ndarray:
        return arr

    @staticmethod
    def convolve_multi_1d(arr: np.ndarray, kernels: Tuple[np.ndarray, ...], axes: Tuple[int, ...]) -> np.ndarray:
        assert len(kernels) == len(axes)

        convolved = arr
        for a, kernel in zip(axes, kernels):
            convolved = convolve1d(convolved, kernel, axis=a, mode='constant', cval=0.0)

        return convolved
