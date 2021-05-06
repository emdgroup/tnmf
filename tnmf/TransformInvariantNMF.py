"""
Transform-Invariant Non-Negative Matrix Factorization
Authors: Adrian Sosic, Mathias Winkel
"""

# TODO: backend-specific return types
# TODO: no atom_shape for non-ShiftInvariantNMF
# TODO: handling of transform input/output shapes
# TODO: naming convention: energy (instead of cost/error)
# TODO: add options for tensor renormalization
# TODO: cache reconstruction result
# TODO: flexible input types for V
# TODO: we extract .shape[...] too often
# TODO: add support for inhibition

import logging
from typing import Tuple, Callable

import numpy as np

from .backends.NumPy import NumPy_Backend
from .backends.NumPy_FFT import NumPy_FFT_Backend
from .backends.PyTorch import PyTorch_Backend
from .backends.NumPy_CachingFFT import NumPy_CachingFFT_Backend


class TransformInvariantNMF:

    def __init__(
            self,
            n_atoms: int,
            atom_shape: Tuple[int, ...],
            n_iterations: int = 1000,
            backend: str = 'numpy_fft',
            logger: logging.Logger = None,
            verbose: int = 0,
            **kwargs,
    ):
        self.atom_shape = atom_shape
        self.n_atoms = n_atoms
        self.n_iterations = n_iterations
        self._axes_W_normalization = tuple(range(-len(atom_shape), 0))
        self.eps = 1.e-9

        backend_map = {
            'numpy': NumPy_Backend,
            'numpy_fft': NumPy_FFT_Backend,
            'numpy_caching_fft': NumPy_CachingFFT_Backend,
            'pytorch': PyTorch_Backend,
        }

        self._backend = backend_map[backend.lower()](**kwargs)

        self._logger = logger if logger is not None else logging.getLogger(self.__class__.__name__)
        self._logger.setLevel([logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG][verbose])

        self.n_iterations_done = 0
        self._W = None
        self._H = None

    @property
    def W(self) -> np.ndarray:
        return self._backend.to_ndarray(self._W)

    @property
    def H(self) -> np.ndarray:
        return self._backend.to_ndarray(self._H)

    @property
    def R(self) -> np.ndarray:
        return self._backend.to_ndarray(self._reconstruct())

    def _reconstruct(self) -> np.ndarray:
        return self._backend.reconstruct(self._W, self._H)

    def _energy_function(self, V: np.ndarray) -> float:
        return self._backend.reconstruction_energy(V, self._W, self._H)

    def _multiplicative_update(self, arr: np.ndarray, neg, pos, sparsity: float = 0):
        assert sparsity >= 0

        regularization = self.eps

        if sparsity > 0:
            regularization += sparsity

        pos += regularization

        arr *= neg
        arr /= pos

    def _update_W(self, V: np.ndarray):
        neg, pos = self._backend.reconstruction_gradient_W(V, self._W, self._H)
        self._multiplicative_update(self._W, neg, pos)
        self._backend.normalize(self._W, axis=self._axes_W_normalization)

    def _update_H(self, V: np.ndarray, sparsity: float = 0):
        neg, pos = self._backend.reconstruction_gradient_H(V, self._W, self._H)
        self._multiplicative_update(self._H, neg, pos, sparsity)

    def _do_fit(
            self,
            V: np.ndarray,
            update_H: bool,
            update_W: bool,
            sparsity_H: float,
            keep_W: bool,
            progress_callback: Callable[['TransformInvariantNMF', int], bool],
    ):
        assert update_H or update_W

        self._W, self._H = self._backend.initialize(
            V, self.atom_shape, self.n_atoms, self._W if keep_W else None)

        if not keep_W:
            self._backend.normalize(self._W, self._axes_W_normalization)

        for self.n_iterations_done in range(self.n_iterations):
            if update_H:
                self._update_H(V, sparsity_H)

            if update_W:
                self._update_W(V)

            if progress_callback is not None:
                if not progress_callback(self, self.n_iterations_done):
                    break
            else:
                self._logger.info(f"Iteration: {self.n_iterations_done}\tEnergy function: {self._energy_function(V)}")

        self._logger.info("NMF finished.")

    def fit(
            self,
            V: np.ndarray,
            update_H: bool = True,
            update_W: bool = True,
            sparsity_H: float = 0.1,
            progress_callback: Callable[['TransformInvariantNMF', int], bool] = None,
    ):
        self._do_fit(V, update_H, update_W, sparsity_H, False, progress_callback)

    def partial_fit(
            self,
            V: np.ndarray,
            update_H: bool = True,
            update_W: bool = True,
            sparsity_H: float = 0.1,
            progress_callback: Callable[['TransformInvariantNMF', int], bool] = None,
    ):
        self._do_fit(V, update_H, update_W, sparsity_H, self.n_iterations_done > 0, progress_callback)
