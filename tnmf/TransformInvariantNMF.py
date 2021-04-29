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

import numpy as np

import logging
from typing import Tuple, Callable
from .backends.NumPy_FFT import NumPy_FFT_Backend
from .backends.PyTorch import PyTorch_Backend


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
        self._axes_W_normalization = tuple(range(1, len(atom_shape)+1))
        self.eps = 1.e-9

        backend_map = {
            'numpy_fft': NumPy_FFT_Backend,
            'pytorch': PyTorch_Backend,
        }

        self._backend = backend_map[backend.lower()](**kwargs)

        self._logger = logger if logger is not None else logging.getLogger(self.__class__.__name__)
        self._logger.setLevel([logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG][verbose])

        self.n_iterations_done = 0

    def reconstruct(self) -> np.ndarray:
        return self._backend.reconstruct(self._W, self._H)

    def _energy_function(self, V: np.ndarray) -> float:
        return self._backend.reconstruction_energy(V, self._W, self._H)

    def _update_W(self, V: np.ndarray):
        neg, pos = self._backend.reconstruction_gradient_W(V, self._W, self._H)

        self._W = self._W * (neg / (pos + self.eps))
        self._W = self._backend.normalize(self._W, axes=self._axes_W_normalization)

    def _update_H(self, V: np.ndarray, sparsity: float = 0):
        assert sparsity >= 0

        neg, pos = self._backend.reconstruction_gradient_H(V, self._W, self._H)

        if sparsity > 0:
            pos = pos + sparsity

        self._H = self._H * (neg / (pos + self.eps))

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

        self._W, self._H = self._backend.initialize_matrices(
            V, self.atom_shape, self.n_atoms, self._W if keep_W else None)

        if not keep_W:
            self._W = self._backend.normalize(self._W, self._axes_W_normalization)

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
