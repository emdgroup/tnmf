"""
Test the decomposition on multiple image patches that are supplied
to the method via an iterator object as representative for streaming
data.
Reconstruction of the final subsample is verified against pre-computed
reference values.
"""

import logging

import numpy as np
import pytest

from tnmf.TransformInvariantNMF import TransformInvariantNMF, MiniBatchAlgorithm
from tnmf.utils.data_loading import racoon_image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

# hard-coded expected energy levels for the different algorithms
expected_energies = {
    # no need to test all algorithms here
    # MiniBatchAlgorithm.Cyclic_MU: 136.8409655,
    # MiniBatchAlgorithm.ASG_MU: 97.0072791,
    # MiniBatchAlgorithm.GSG_MU: 136.43285833,
    MiniBatchAlgorithm.ASAG_MU: 96.7375921,
    # MiniBatchAlgorithm.GSAG_MU: 136.7082644,
}

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


def _do_test(samples, algorithm):

    # use the same random seed for all runs
    np.random.seed(seed=42)

    # create and fit the NMF model
    nmf = TransformInvariantNMF(
        n_atoms=10,
        atom_shape=(7, 7),
        backend='numpy_caching_fft',
        verbose=3,
        reconstruction_mode='valid',
    )
    nmf.fit(
        samples,
        sparsity_H=0.1,
        algorithm=algorithm,
        subsample_size=50,
        batch_size=3,
        n_epochs=5,
        sag_lambda=0.8)

    # extract the reconstruction mode dependent expected energy level
    expected_energy = expected_energies[algorithm]

    # check if the expected energy level is reached
    assert np.isclose(nmf._energy_function(), expected_energy)  # pylint: disable=protected-access

    # check if the atoms have unit norm
    norm_W = np.sum(nmf.W, axis=(-1, -2))
    assert np.allclose(norm_W, 1.)


@pytest.mark.parametrize('algorithm', list(expected_energies.keys()))
def test_with_array(algorithm: int):
    _do_test(V, algorithm)


@pytest.mark.parametrize('algorithm', list(expected_energies.keys()))
def test_with_generator(algorithm: int):
    _do_test((v for v in V), algorithm)


def test_with_generator_limited():
    # use the same random seed for all runs
    np.random.seed(seed=42)

    # create and fit the NMF model
    nmf = TransformInvariantNMF(
        n_atoms=10,
        atom_shape=(7, 7),
        backend='numpy_caching_fft',
        verbose=3,
        reconstruction_mode='valid',
    )
    nmf.fit(
        V,
        sparsity_H=0.1,
        algorithm=MiniBatchAlgorithm.Cyclic_MU,
        subsample_size=50,
        max_subsamples=5,
        batch_size=3,
        n_epochs=5,
        sag_lambda=0.8)

    # check if the expected energy level is reached
    assert np.isclose(nmf._energy_function(), 629.109136)  # pylint: disable=protected-access
