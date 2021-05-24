"""
Test the decomposition on two identical images and ensure that all backends yield the same factorization.
"""

import logging
from typing import Dict

import numpy as np
import pytest

from tnmf.TransformInvariantNMF import TransformInvariantNMF
from tnmf.tests.utils import racoon_image

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s: %(message)s')

# define all test settings
backends = ['numpy', 'numpy_fft', 'numpy_caching_fft', 'pytorch']
# hard-coded expected energy levels for the different reconstruction modes
test_params = [
    # increasing sparsity, no inhibition
    (dict(sparsity_H=0.0, inhibition_strength=0.0), dict(energy=186.666013, norm_H_1=7704.38977, norm_H_0=176346)),
    (dict(sparsity_H=0.1, inhibition_strength=0.0), dict(energy=225.494731, norm_H_1=6563.91176, norm_H_0=174037)),
    (dict(sparsity_H=0.5, inhibition_strength=0.0), dict(energy=858.621063, norm_H_1=4258.82541, norm_H_0=155247)),
    (dict(sparsity_H=1.0, inhibition_strength=0.0), dict(energy=2429.69334, norm_H_1=2114.50047, norm_H_0=136396)),
    (dict(sparsity_H=5.0, inhibition_strength=0.0), dict(energy=5351.91865, norm_H_1=3.0800e-06, norm_H_0=65338)),
    (dict(sparsity_H=10., inhibition_strength=0.0), dict(energy=5351.91866, norm_H_1=2.5103e-13, norm_H_0=62486)),
    # no sparsity, increasing inhibition
    (dict(sparsity_H=0.0, inhibition_strength=0.1), dict(energy=234.838968, norm_H_1=6730.89543, norm_H_0=176347)),
    (dict(sparsity_H=0.0, inhibition_strength=0.5), dict(energy=680.585424, norm_H_1=5177.87844, norm_H_0=174277)),
    (dict(sparsity_H=0.0, inhibition_strength=1.0), dict(energy=1119.00855, norm_H_1=4657.19574, norm_H_0=168777)),
    (dict(sparsity_H=0.0, inhibition_strength=5.0), dict(energy=518.936361, norm_H_1=6872.57858, norm_H_0=100488)),
    (dict(sparsity_H=0.0, inhibition_strength=10.), dict(energy=489.935256, norm_H_1=7224.76002, norm_H_0=62017)),
]

# create the input by concatenating the test image twice
img = racoon_image(gray=False, scale=0.1)
V = np.repeat(img.transpose((2, 0, 1))[np.newaxis, ...], 2, axis=0)


def fit_nmf(backend, params):
    # use the same random seed for all runs
    np.random.seed(seed=42)

    # create and fit the NMF model
    nmf = TransformInvariantNMF(
        n_atoms=10,
        atom_shape=(7, 7),
        n_iterations=25,
        backend=backend,
        verbose=3,
    )
    nmf.fit(V, **params)

    return nmf


@pytest.mark.parametrize('params', test_params)
@pytest.mark.parametrize('backend', backends)
def test_expected_energy(backend: str, params: Dict):

    # fit the NMF model
    nmf = fit_nmf(backend, params[0])
    H = nmf.H

    energy = nmf._energy_function(V)  # pylint: disable=protected-access
    norm_H_1 = np.sum(np.abs(H))
    norm_H_0 = np.sum(H/H.max() > 1e-7)

    nmf._logger.debug(f'energy={energy}, norm_H_1={norm_H_1}, norm_H_0={norm_H_0}')  # pylint: disable=protected-access

    expectation = params[1]
    assert np.isclose(energy, expectation['energy'])
    assert np.isclose(norm_H_1, expectation['norm_H_1'])
    assert np.isclose(norm_H_0, expectation['norm_H_0'])
