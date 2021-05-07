"""
Test the decomposition on two identical images.
"""
import logging

import numpy as np

from tnmf.old.TransformInvariantNMF import ShiftInvariantNMF
from tnmf.tests.utils import racoon_image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')


def do_test(use_fft: bool, expected_error: float):
    img = racoon_image(gray=False, scale=0.1)
    V = np.repeat(img[..., None], 2, axis=-1)

    nmf = ShiftInvariantNMF(
            atom_size=7,
            n_components=10,
            sparsity_H=0.1,
            refit_H=False,
            n_iterations=10,
            eps=1e-9,
            logger=None,
            verbose=3,
            inhibition_range=None,
            inhibition_strength=0.,
            use_fft=use_fft
        )

    nmf.fit(V)

    assert np.isclose(nmf.reconstruction_error(), expected_error)

    assert np.isclose(0.5 * np.sum(np.square(nmf.R - V)), expected_error)


def test_numpy():
    np.random.seed(seed=42)
    do_test(use_fft=False, expected_error=268.14423)


def test_numpy_fft():
    np.random.seed(seed=42)
    do_test(use_fft=False, expected_error=268.14423)
