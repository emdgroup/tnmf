"""
Test the decomposition on two identical images and ensure that all backends arrive at the same energy level.
"""

import logging
import pytest

import numpy as np

from tnmf.TransformInvariantNMF import TransformInvariantNMF
from tnmf.tests.utils import racoon_image
from itertools import product

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


@pytest.mark.parametrize('backend, reconstruction_mode', product(backends, reconstruction_modes))
def test_expected_energy(backend: str, reconstruction_mode: str):

    # use the same random seed for all runs
    np.random.seed(seed=42)

    # extract the reconstruction mode dependent expected energy level
    expected_energy = expected_energies[reconstruction_mode]

    # create the input by concatenating the test image twice
    img = racoon_image(gray=False, scale=0.1)
    V = np.repeat(img.transpose((2, 0, 1))[np.newaxis, ...], 2, axis=0)

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
    except NotImplementedError:
        if raise_not_implement_errors:
            raise AssertionError
        else:
            return
    nmf.fit(V)

    # check if the expected energy level is reached
    assert np.isclose(nmf._energy_function(V), expected_energy)  # pylint: disable=protected-access

    # check if the atoms have unit norm
    norm_W = np.sum(nmf.W, axis=(-1, -2))
    assert np.allclose(norm_W, 1.)
