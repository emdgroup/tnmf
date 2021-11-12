"""
A module that provides a NumPy based backend for computing the gradients of the factorization model.
Shift-invariance is implemented via fast convolution in the Fourier domain using :func:`scipy.fft.rfftn`
and :func:`scipy.fft.irfftn`.
"""
from typing import Tuple, Dict

import numpy as np
from opt_einsum.contract import ContractExpression
from scipy.fft import rfftn, irfftn

from ._Backend import sliceNone
from ._NumPyFFTBackend import NumPyFFTBackend


def _fft_convolve(
    arr1: Tuple[np.ndarray, ...],
    arr2: np.ndarray,
    contraction: ContractExpression,
    slices: Tuple[slice, ...],
    fft_axes: Tuple[int, ...],
    fft_shape: Tuple[int, ...],
    pad_mode: Dict = None,
    pad_width: Tuple[Tuple[int, int], ...] = None,
    correlate: bool = False,
    fft_workers: int = -1,
) -> Tuple[np.ndarray, ...]:

    c2 = arr2 if not correlate else np.flip(arr2, axis=fft_axes)
    f2 = rfftn(c2, axes=fft_axes, s=fft_shape, workers=fft_workers)

    ret = tuple()
    for arr in arr1:
        c1 = arr if pad_mode is None else np.pad(arr, pad_width, **pad_mode)
        f1 = rfftn(c1, axes=fft_axes, s=fft_shape, workers=fft_workers)

        fr = contraction(f1, f2)
        cr = irfftn(fr, axes=fft_axes, s=fft_shape, workers=fft_workers)[slices]
        ret += (cr.copy(), )
    return ret


class NumPy_FFT_Backend(NumPyFFTBackend):
    r"""
    A NumPy based backend that performs convolutions and contractions for computing the gradients of the factorization model
    via FFT:

    Arrays to be convolved are transformed to Fourier space, multiplied and accumulated across the free indices (e.g. for the
    sum over all atoms in the reconstruction), and transformed back to coordinate space.
    """

    def reconstruction_gradient_W(
        self,
        V: np.ndarray,
        W: np.ndarray,
        H: np.ndarray,
        s: slice = sliceNone
    ) -> Tuple[np.ndarray, np.ndarray]:

        R = self.reconstruct(W, H[s])
        assert R.shape == V[s].shape

        #        sum_n  H[n , m, ...] * V|R[n, c, ... ] --> dR / dW[m, c, ...]
        neg, pos = _fft_convolve((V[s], R), H[s], **self.fft_params['grad_W'])

        assert neg.shape == W.shape
        assert pos.shape == W.shape

        return neg, pos

    def reconstruction_gradient_H(
        self,
        V: np.ndarray,
        W: np.ndarray,
        H: np.ndarray,
        s: slice = sliceNone
    ) -> Tuple[np.ndarray, np.ndarray]:

        R = self.reconstruct(W, H[s])
        assert R.shape == V[s].shape

        #        sum_c  V|R[n, c, ... ] * W[m, c, ...] --> dR / dH[n, m, ...]
        neg, pos = _fft_convolve((V[s], R), W, **self.fft_params['grad_H'])

        assert neg.shape == H[s].shape
        assert pos.shape == H[s].shape

        return neg, pos

    def reconstruct(self, W: np.ndarray, H: np.ndarray) -> np.ndarray:
        #        sum_m  H[n, m, ... ] * W[m, c, ...] --> R[n, c, ...]
        R, = _fft_convolve((H, ), W, **self.fft_params['reconstruct'])
        return R
