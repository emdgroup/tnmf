"""
Test the decomposition on multiple image patches in minibatch mode
and verify that all backends yield the same factorization.
"""

import logging
from typing import Tuple

import numpy as np
import pytest

from tnmf.TransformInvariantNMF import TransformInvariantNMF, MiniBatchAlgorithm
from tnmf.utils.data_loading import racoon_image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

# hard-coded expected energy levels for the different algorithms
expected_energies = {
    'full_batch': 14434.02658,
    MiniBatchAlgorithm.Cyclic_MU: 14434.02658,
    MiniBatchAlgorithm.ASG_MU: 4558.86695,
    MiniBatchAlgorithm.GSG_MU: 14223.14454,
    MiniBatchAlgorithm.ASAG_MU: 4560.03432,
    MiniBatchAlgorithm.GSAG_MU: 14310.92041,
}

# define all test settings
backends = [
    'numpy',
    'numpy_fft',
    'numpy_caching_fft',
    'pytorch',
    'pytorch_fft']

img = racoon_image(gray=True, scale=1.)
# creative way of extracting image blocks (attention: patch_shape must divide img.shape)
shape = np.array(img.shape*2)
image_shape = np.asarray(img.shape*2)
patch_shape = np.array([32, 32])
image_shape[:2] = image_shape[:2] / patch_shape
image_shape[2:] = patch_shape
byte_size = img.strides[-1]  # pylint: disable=unsubscriptable-object
image_strides = np.array([img.shape[0]*patch_shape[1], patch_shape[0], img.shape[0], 1]) * byte_size
V = np.lib.stride_tricks.as_strided(img, shape=image_shape, strides=image_strides)
V = V.reshape((-1, 1, *V.shape[2:]))


def fit_nmf(backend, algorithm):
    # use the same random seed for all runs
    np.random.seed(seed=42)

    # create and fit the NMF model
    nmf = TransformInvariantNMF(
        n_atoms=10,
        atom_shape=(7, 7),
        backend=backend,
        verbose=3,
        reconstruction_mode='valid',
    )
    if isinstance(algorithm, MiniBatchAlgorithm):
        nmf.fit_minibatches(
            V,
            sparsity_H=0.1,
            algorithm=algorithm,
            batch_size=3,
            n_epochs=5,
            sag_lambda=0.8)
    else:
        nmf.fit_batch(
            V,
            sparsity_H=0.1,
            n_iterations=5,
        )

    return nmf


@pytest.fixture(name='expected_factorization')
def fixture_expected_factorization(algorithm):
    nmf = fit_nmf('pytorch', algorithm)
    return nmf.W, nmf.H, nmf.R, nmf.R_partial(0)


@pytest.mark.parametrize('algorithm', list(expected_energies.keys()))
@pytest.mark.parametrize('backend', backends)
def test_expected_energy(backend: str, algorithm: int, expected_factorization: Tuple[np.ndarray, np.ndarray]):

    # extract the target tensors and the reconstruction mode dependent expected energy level
    W, H, R, R0 = expected_factorization
    expected_energy = expected_energies[algorithm]

    # fit the NMF model
    nmf = fit_nmf(backend, algorithm)

    # check if the expected energy level is reached
    assert np.isclose(nmf._energy_function(), expected_energy)  # pylint: disable=protected-access

    # check if the expected factorization is obtained
    assert np.allclose(nmf.W, W)
    assert np.allclose(nmf.H, H)
    assert np.allclose(nmf.R, R)
    assert np.allclose(nmf.R_partial(0), R0)

    # check if the atoms have unit norm
    norm_W = np.sum(nmf.W, axis=(-1, -2))
    assert np.allclose(norm_W, 1.)
