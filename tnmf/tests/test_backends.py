"""
Test the decomposition on two identical images and ensure that all backends yield the same factorization.
"""

import logging
import pytest

import numpy as np

from tnmf.TransformInvariantNMF import TransformInvariantNMF
from tnmf.tests.utils import racoon_image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

# hard-coded expected energy levels for the different reconstruction modes
expected_energies = {
    'valid': 268.14423,
}

# define all test settings
backends = ['numpy', 'numpy_fft', 'numpy_caching_fft', 'pytorch']
reconstruction_modes = ['valid']

# temporarily ignore failed tests due to unimplemented features
raise_not_implement_errors = False

# create the input by concatenating the test image twice
img = racoon_image(gray=False, scale=0.1)
V = np.repeat(img.transpose((2, 0, 1))[np.newaxis, ...], 2, axis=0)


def fit_nmf(backend, reconstruction_mode):
    # use the same random seed for all runs
    np.random.seed(seed=42)

    # create and fit the NMF model
    try:
        nmf = TransformInvariantNMF(
            n_atoms=10,
            atom_shape=(7, 7),
            n_iterations=10,
            backend=backend,
            verbose=3,
            reconstruction_mode=reconstruction_mode,
        )
    except NotImplementedError as e:
        if raise_not_implement_errors:
            raise AssertionError from e
    nmf.fit(V)

    return nmf


@pytest.fixture(name='expected_factorization')
def fixture_expected_factorization():
    nmf = fit_nmf('numpy', 'valid')
    return nmf.W, nmf.H


@pytest.mark.parametrize('reconstruction_mode', reconstruction_modes)
@pytest.mark.parametrize('backend', backends)
def test_expected_energy(backend: str, reconstruction_mode: str, expected_factorization):

    # extract the target tensors and the reconstruction mode dependent expected energy level
    W, H = expected_factorization
    expected_energy = expected_energies[reconstruction_mode]

    # fit the nmf model
    nmf = fit_nmf(backend, reconstruction_mode)

    # check if the expected energy level is reached
    assert np.isclose(nmf._energy_function(V), expected_energy)  # pylint: disable=protected-access

    # check if the expected factorization is obtained
    assert np.allclose(nmf.W, W)
    assert np.allclose(nmf.H, H)

    # check if the atoms have unit norm
    norm_W = np.sum(nmf.W, axis=(-1, -2))
    assert np.allclose(norm_W, 1.)
