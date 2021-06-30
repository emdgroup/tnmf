"""
A module that provides a NumPy based backend for computing the gradients of the factorization model.
Shift-invariance is implemented via fast convolution in the Fourier domain using :func:`scipy.fft.rfftn`
and :func:`scipy.fft.irfftn`.
"""
from typing import Tuple, Dict, Union

import numpy as np
from scipy.fft import next_fast_len, rfftn, irfftn

from ._NumPyFFTBackend import NumPyFFTBackend


def fftconvolve_sum(
        in1: Union[np.ndarray, Tuple[np.ndarray, ...]],
        in2: np.ndarray,
        fft_axes: Tuple[int, ...],
        slices: Tuple[slice, ...],
        correlate: bool,
        pad_mode: Dict = None,
        pad_width: Tuple[Tuple[int, int], ...] = None,
        sum_axis: Tuple[int, ...] = None,
        keepdims: bool = False,
) -> np.ndarray:

    def padded_rfftn(
        array: np.ndarray,
        fft_shape: Tuple[int, ...],
        axes: Tuple[int, ...],
        pad_mode: Dict = None,
        pad_shape: Dict = None
    ) -> np.ndarray:
        # padding to fulfill boundary conditions
        if pad_mode is not None:
            array = np.pad(array, pad_shape, **pad_mode)

        # first axes are not to be padded
        fftpadding = tuple(((0, 0), ) * (array.ndim - len(axes)))
        # last axes will be padded
        fftpadding += tuple((0, fft_shape[ia] - array.shape[a]) for ia, a in enumerate(axes))
        # zero-pad the relevant axes
        array_padded = np.pad(array, fftpadding, mode='constant', constant_values=0.)
        # Fourier transform (for real data)
        return rfftn(array_padded, axes=axes)

    if not isinstance(in1, tuple):
        assert isinstance(in1, np.ndarray)
        in1 = (in1, )

    assert isinstance(in2, np.ndarray)

    ndim = in2.ndim
    s1 = in1[0].shape
    s2 = in2.shape

    assert all(a.ndim == ndim for a in in1)
    assert all(a.shape == s1 for a in in1)

    # convert negative axes indices into positive counterparts
    axes = sorted([a if a >= 0 else a + ndim for a in fft_axes])

    # pad according to required boundary conditions
    if pad_mode is not None:
        # we need padding information for all axes to be convolved over
        assert pad_width is not None and len(pad_width) == len(axes)

        pad_shape = tuple(((0, 0), ) * (ndim - len(axes))) + pad_width
        padded_shape = np.asarray(pad_shape).sum(axis=1) + np.asarray(s1)
    else:
        pad_shape = None
        padded_shape = s1

    # now need to zero-pad to identical size / optimal fft size
    fft_shape = tuple(next_fast_len(padded_shape[a] + s2[a] - 1, True) for a in axes)

    # compute how to undo the padding
    fslice = (slice(None), ) * (ndim - len(axes)) + slices

    # for correlation instead of convolution, we simply reverse the second array
    if correlate:
        reverse = ((slice(None), )) * (ndim - len(fft_axes)) + (slice(None, None, -1),) * len(fft_axes)
        assert len(reverse) == len(s2)
        in2 = in2[reverse]

    # Fourier transform (for real data)
    sp1 = (padded_rfftn(in1i, fft_shape, axes, pad_mode, pad_shape) for in1i in in1)
    sp2 = padded_rfftn(in2, fft_shape, axes)

    result = tuple()

    for sp1i in sp1:
        # perform convolution by multiplication in Fourier space
        sp1sp2 = sp1i * sp2

        # sum over an axis if requested, set the corresponding dimension to 1
        if sum_axis is not None:
            sp1sp2 = np.sum(sp1sp2, axis=sum_axis, keepdims=True)

        # transform back to real space
        ret = np.asarray(irfftn(sp1sp2, axes=axes))

        # actually remove the padded rows/columns
        ret = ret[fslice].copy()

        # remove singleton dimensions if requested
        if not keepdims and sum_axis is not None:
            assert ret.shape[sum_axis] == 1
            ret = np.squeeze(ret, sum_axis)

        result += (ret, )

    return result


class NumPy_FFT_Backend(NumPyFFTBackend):
    r"""
    A NumPy based backend that performs convolutions and contractions for computing the gradients of the factorization model
    via FFT:

    Arrays to be convolved are transformed to Fourier space, multiplied and accumulated across the free indices (e.g. for the
    sum over all atoms in the reconstruction), and transformed back to coordinate space.
    """

    def reconstruction_gradient_W(self, V: np.ndarray, W: np.ndarray, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        R = self.reconstruct(W, H)
        assert R.shape == V.shape

        #          sum_n    H[n , m, _, ...] * V|R[n, _, c, ... ]   --> dR / dW[m, c, ...]
        neg, pos = fftconvolve_sum(
            (V[:, np.newaxis, :, ...], R[:, np.newaxis, :, ...]),
            H[:, :, np.newaxis, ...],
            sum_axis=0,
            **self.fft_params['grad_W'])

        assert neg.shape == W.shape
        assert pos.shape == W.shape

        return neg, pos

    def reconstruction_gradient_H(self, V: np.ndarray, W: np.ndarray, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        R = self.reconstruct(W, H)
        assert R.shape == V.shape

        #        sum_c        V|R[n, _, c, ... ] * W[_ , m, c, ...]   --> dR / dH[n, m, ...]
        neg, pos = fftconvolve_sum(
            (V[:, np.newaxis, :, ...], R[:, np.newaxis, :, ...]),
            W[np.newaxis, :, :, ...],
            sum_axis=2,
            **self.fft_params['grad_H'])

        assert neg.shape == H.shape
        assert pos.shape == H.shape

        return neg, pos

    def reconstruct(self, W: np.ndarray, H: np.ndarray) -> np.ndarray:
        #        sum_m    H[n, m, _, ... ] * W[_ , m, c, ...]   --> R[n, c, ...]
        R, = fftconvolve_sum(
            (H[:, :, np.newaxis, ...], ),
            W[np.newaxis, :, ...],
            sum_axis=1,
            **self.fft_params['reconstruct'])

        return R
