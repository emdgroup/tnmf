"""
Test if a single sample decomposition works
"""

import logging

import numpy as np
import torch
from scipy.misc import face

from tnmf.TransformInvariantNMF import TransformInvariantNMF

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')


def do_test(backend: str, expected_error: float):
    img = face(gray=True) / 255

    # downsample for higher speed
    img = img[::4, ::4] + img[1::4, ::4] + img[::4, 1::4] + img[1::4, 1::4]
    img /= 4.0

    img = img[np.newaxis, np.newaxis, ...]

    patch_shape = (7, 7)

    nmf = TransformInvariantNMF(
        n_atoms=10,
        atom_shape=patch_shape,
        n_iterations=25,
        backend=backend,
        logger=None,
        verbose=3,
    )

    nmf.fit(img)

    assert np.isclose(nmf._energy_function(img), expected_error)

    img_r = nmf.reconstruct()
    assert np.isclose(0.5*np.sum(np.square(img_r - img)), expected_error)


def test_numpy():
    np.random.seed(seed=42)
    do_test('numpy', 104.74284)


def test_numpy_fft():
    np.random.seed(seed=42)
    do_test('numpy_fft', 104.74284)


def test_numpy_caching_fft():
    np.random.seed(seed=42)
    do_test('numpy_caching_fft', 104.74284)


def test_pytorch():
    torch.manual_seed(42)
    do_test('pytorch', 104.2733)
