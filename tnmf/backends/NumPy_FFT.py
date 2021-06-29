"""
A module that provides a NumPy based backend for computing the gradients of the factorization model.
Shift-invariance is implemented via fast convolution in the Fourier domain using :func:`scipy.fft.rfftn`
and :func:`scipy.fft.irfftn`.
"""
from typing import Tuple, Dict, Union

import numpy as np
from scipy.fft import next_fast_len, rfftn, irfftn

from ._NumPyBackend import NumPyBackend


def fftconvolve_sum(
        in1: Union[np.ndarray, Tuple[np.ndarray, ...]],
        in2: Union[np.ndarray, Tuple[np.ndarray, ...]],
        convolve_axes: Tuple[int, ...],
        output_lower_index: Tuple[int, ...],
        output_shape: Tuple[int, ...],
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

    if not isinstance(in2, tuple):
        assert isinstance(in2, np.ndarray)
        in2 = (in2, )

    ndim = in1[0].ndim
    s1 = in1[0].shape
    s2 = in2[0].shape

    assert all(a.ndim == ndim for a in in1 + in2)
    assert all(a.shape == s1 for a in in1)
    assert all(a.shape == s2 for a in in2)

    # convert negative axes indices into positive counterparts
    axes = sorted([a if a >= 0 else a + ndim for a in convolve_axes])

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
    fslice = (slice(None), ) * (ndim - len(axes))
    fslice += tuple(slice(f, f + s) for f, s in zip(output_lower_index, output_shape))

    # Fourier transform (for real data)
    sp1 = tuple(padded_rfftn(in1i, fft_shape, axes, pad_mode, pad_shape) for in1i in in1)
    sp2 = tuple(padded_rfftn(in2i, fft_shape, axes) for in2i in in2)

    result = tuple()

    for sp1i in sp1:
        for sp2i in sp2:
            # perform convolution by multiplication in Fourier space
            sp1sp2 = sp1i * sp2i

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


class NumPy_FFT_Backend(NumPyBackend):
    r"""
    A NumPy based backend that performs convolutions and contractions for computing the gradients of the factorization model
    via FFT:

    Arrays to be convolved are transformed to Fourier space, multiplied and accumulated across the free indices (e.g. for the
    sum over all atoms in the reconstruction), and transformed back to coordinate space.
    """

    def reconstruction_gradient_W(self, V: np.ndarray, W: np.ndarray, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        R = self.reconstruct(W, H)
        assert R.shape == V.shape

        # do not reverse n_samples and n_atoms dimension
        reverse = (slice(None, None, 1),) * 2 + (slice(None, None, -1),) * self._n_shift_dimensions
        assert len(reverse) == H.ndim
        H_reversed = H[reverse]

        #          sum_n    H[n , m, _, ...] * V|R[n, _, c, ... ]   --> dR / dW[m, c, ...]
        neg, pos = fftconvolve_sum(
            H_reversed[:, :, np.newaxis, ...],
            (V[:, np.newaxis, :, ...], R[:, np.newaxis, :, ...]),
            sum_axis=0,
            output_lower_index=np.minimum(np.array(self._sample_shape), np.array(self._transform_shape)) - 1,
            output_shape=self.atom_shape,
            pad_mode=self._pad_mode,
            pad_width=self._padding_right,
            convolve_axes=self._shift_dimensions)

        assert neg.shape == W.shape
        assert pos.shape == W.shape

        return neg, pos

    def reconstruction_gradient_H(self, V: np.ndarray, W: np.ndarray, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        R = self.reconstruct(W, H)
        assert R.shape == V.shape

        # do not reverse n_atoms and n_channels dimension
        reverse = (slice(None, None, 1),) * 2 + (slice(None, None, -1),) * self._n_shift_dimensions
        assert len(reverse) == W.ndim
        W_reversed = W[reverse]

        #        sum_c        V|R[n, _, c, ... ] * W[_ , m, c, ...]   --> dR / dH[n, m, ...]
        neg, pos = fftconvolve_sum(
            (V[:, np.newaxis, :, ...], R[:, np.newaxis, :, ...]),
            W_reversed[np.newaxis, :, :, ...],
            sum_axis=2,
            output_lower_index=np.asarray(self._padding_right)[:, 1] if self._pad_mode is not None else np.zeros_like(self._transform_shape),
            output_shape=self._transform_shape,
            pad_mode=self._pad_mode,
            pad_width=self._padding_right,
            convolve_axes=self._shift_dimensions)

        assert neg.shape == H.shape
        assert pos.shape == H.shape

        return neg, pos

    def reconstruct(self, W: np.ndarray, H: np.ndarray) -> np.ndarray:
        #        sum_m    H[n, m, _, ... ] * W[_ , m, c, ...]   --> R[n, c, ...]
        R, = fftconvolve_sum(
            H[:, :, np.newaxis, ...],
            W[np.newaxis, :, ...],
            sum_axis=1,
            output_lower_index=np.array(self.atom_shape) - 1,
            output_shape=self._sample_shape,
            pad_mode=self._pad_mode,
            pad_width=self._padding_left,
            convolve_axes=self._shift_dimensions)

        return R
