"""
Test if a single sample decomposition works
"""
import logging

import numpy as np
from scipy.misc import face

from tnmf.old.TransformInvariantNMF import ShiftInvariantNMF

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')


def do_test(use_fft: bool, expected_error: float):
    img = face(gray=True) / 255

    # downsample for higher speed
    img = img[::4, ::4] + img[1::4, ::4] + img[::4, 1::4] + img[1::4, 1::4]
    img /= 4.0

    img = img[..., np.newaxis, np.newaxis]

    patch_shape = 7

    nmf = ShiftInvariantNMF(
        atom_size=patch_shape,
        n_components=10,
        sparsity_H=0.1,
        refit_H=False,
        n_iterations=25,
        eps=1e-9,
        logger=None,
        verbose=3,
        inhibition_range=None,
        inhibition_strength=0.,
        use_fft=use_fft
    )

    nmf.fit(img)

    assert np.isclose(nmf.reconstruction_error(), expected_error)

    img_r = nmf.R
    assert np.isclose(0.5 * np.sum(np.square(img_r - img)), expected_error)


def test_numpy():
    np.random.seed(seed=42)
    do_test(use_fft=False, expected_error=339.97078)


def test_numpy_fft():
    np.random.seed(seed=42)
    do_test(use_fft=False, expected_error=339.97078)
