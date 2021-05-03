from typing import Tuple, Dict

import numpy as np
from scipy.fft import next_fast_len, rfftn, irfftn

from ._NumPyBackend import NumPyBackend


def fftconvolve_sum(
        in1: np.ndarray,
        in2: np.ndarray,
        mode: str = "full",
        axes: Tuple[int, ...] = None,
        sum_axis: Tuple[int, ...] = None,
        padding1: Dict = None,
        padding2: Dict = None,
):

    def _centered(arr, newshape):
        # Return the center newshape portion of the array.
        newshape = np.asarray(newshape)
        currshape = np.array(arr.shape)
        startind = (currshape - newshape) // 2
        endind = startind + newshape
        myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
        return arr[tuple(myslice)]

    def _rfftn_padded(field, padding, fshape, axes):
        pad_width = [((0, fshape[a] - field.shape[a]) if a in axes else (0, 0)) for a in range(field.ndim)]
        field_padded = np.pad(field, pad_width, **padding)
        return rfftn(field_padded, np.array(fshape)[axes], axes=axes)

    if padding1 is None:
        padding1 = dict(mode='constant', constant_values=0)
    if padding2 is None:
        padding2 = dict(mode='constant', constant_values=0)

    assert in1.ndim == in2.ndim
    if axes is None:
        axes = list(range(in1.ndim))

    s1 = list(in1.shape)
    s2 = list(in2.shape)
    axes = sorted([a if a >= 0 else a+in1.ndim for a in axes])
    axes = [a for a in axes if s1[a] != 1 and s2[a] != 1]

    if mode == 'valid':
        if not all(s1[i] >= s2[i] for i in axes):
            in1, in2, s1, s2, padding1, padding2 = in2, in1, s2, s1, padding2, padding1

    shape = [max((s1[i], s2[i])) if i not in axes else s1[i] + s2[i] - 1 for i in range(in1.ndim)]
    fshape = [next_fast_len(shape[a], True) if a in axes else 0 for a in range(in1.ndim)]

    sp1 = _rfftn_padded(in1, padding1, fshape, axes)
    sp2 = _rfftn_padded(in2, padding2, fshape, axes)
    sp1sp2 = sp1 * sp2

    fslice = [slice(sz) for sz in shape]

    if sum_axis is not None:
        if sum_axis < 0:
            sum_axis = sp1sp2.ndim + sum_axis
        assert sum_axis not in axes
        axes = [(a if a < sum_axis else a - 1) for a in axes]
        sp1sp2 = np.sum(sp1sp2, axis=sum_axis)
        del fslice[sum_axis]
        del fshape[sum_axis]
        del s1[sum_axis]
        del s2[sum_axis]

    ret = irfftn(sp1sp2, np.array(fshape)[axes], axes=axes)
    ret = ret[fslice]

    if mode == "full":
        ret = ret.copy()
    elif mode == "same":
        ret = _centered(ret, s1).copy()
    elif mode == "valid":
        shape_valid = [ret.shape[a] if a not in axes else s1[a] - s2[a] + 1 for a in range(ret.ndim)]
        ret = _centered(ret, shape_valid).copy()
    else:
        raise ValueError("acceptable mode flags are 'valid', 'same', or 'full'")

    return ret


class NumPy_FFT_Backend(NumPyBackend):

    def reconstruction_gradient_W(self, V: np.ndarray, W: np.ndarray, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        R = self.reconstruct(W, H)
        assert R.shape == V.shape

        # do not reverse n_samples and n_atoms dimension
        reverse = (slice(None, None, 1),) * 2 + (slice(None, None, -1),) * self._n_shift_dimensions
        assert len(reverse) == H.ndim
        H_reversed = H[reverse]

        #                     V|R[n, _, c, ... ] * H[n , m, _, ...]   --> dR / dW[m, c, ...]
        neg = fftconvolve_sum(
            V[:, np.newaxis, :, ...], H_reversed[:, :, np.newaxis, ...], padding1=self._input_padding,
            padding2=self._input_padding, mode='valid', axes=self._shift_dimensions, sum_axis=0)
        pos = fftconvolve_sum(
            R[:, np.newaxis, :, ...], H_reversed[:, :, np.newaxis, ...], padding1=self._input_padding,
            padding2=self._input_padding, mode='valid', axes=self._shift_dimensions, sum_axis=0)
        return neg, pos

    def reconstruction_gradient_H(self, V: np.ndarray, W: np.ndarray, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        R = self.reconstruct(W, H)
        assert R.shape == V.shape

        # do not reverse n_atoms and n_channels dimension
        reverse = (slice(None, None, 1),) * 2 + (slice(None, None, -1),) * self._n_shift_dimensions
        assert len(reverse) == W.ndim
        W_reversed = W[reverse]

        #        sum_c        V|R[n, _, c, ... ] * W[_ , m, c, ...]   --> dR / dH[n, m, ...]
        neg = fftconvolve_sum(
            V[:, np.newaxis, :, ...], W_reversed[np.newaxis, :, :, ...],
            padding1=self._input_padding, mode=self._mode_H, axes=self._shift_dimensions, sum_axis=2)
        pos = fftconvolve_sum(
            R[:, np.newaxis, :, ...], W_reversed[np.newaxis, :, :, ...],
            padding1=self._input_padding, mode=self._mode_H, axes=self._shift_dimensions, sum_axis=2)
        return neg, pos

    def reconstruct(self, W: np.ndarray, H: np.ndarray) -> np.ndarray:
        #        sum_m    H[n, m, _, ... ] * W[_ , m, c, ...]   --> R[n, c, ...]
        R = fftconvolve_sum(
            H[:, :, np.newaxis, ...], W[np.newaxis, :, ...],
            padding1=self._input_padding, mode=self._mode_R, axes=self._shift_dimensions, sum_axis=1)
        return R
