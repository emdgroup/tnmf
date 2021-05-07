"""
Test the decomposition on two identical images.
"""

import logging

import numpy as np

from tnmf.TransformInvariantNMF import TransformInvariantNMF
from tnmf.tests.utils import racoon_image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')


def do_test(backend: str, expected_error: float):
    img = racoon_image(gray=False, scale=0.1)
    V = np.repeat(img.transpose((2, 0, 1))[None, ...], 2, axis=0)

    nmf = TransformInvariantNMF(
        n_atoms=10,
        atom_shape=(7, 7),
        n_iterations=10,
        backend=backend,
        verbose=3,
    )

    nmf.fit(V)

    assert np.isclose(nmf._energy_function(V), expected_error)  # pylint: disable=protected-access

    assert np.isclose(0.5 * np.sum(np.square(nmf.R - V)), expected_error)

    norm_W = np.sum(nmf.W, axis=(-1, -2))
    assert np.allclose(norm_W, 1.)


def test_numpy():
    np.random.seed(seed=42)
    do_test('numpy', 268.14423)


def test_numpy_fft():
    np.random.seed(seed=42)
    do_test('numpy_fft', 268.14423)


def test_numpy_caching_fft():
    np.random.seed(seed=42)
    do_test('numpy_caching_fft', 268.14423)


def test_pytorch():
    np.random.seed(seed=42)
    do_test('pytorch', 268.14423)
