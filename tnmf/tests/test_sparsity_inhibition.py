"""
Test the decomposition on two identical images with different values for sparsity and inhibition and
ensure that all backends yield the same reconstruction energy and L0 and L1 norm for the activations H
"""

import logging
from typing import Dict

import numpy as np
import pytest

from tnmf.TransformInvariantNMF import TransformInvariantNMF
from tnmf.utils.data_loading import racoon_image

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s: %(message)s')

# define all test settings
backends = ['numpy', 'numpy_fft', 'numpy_caching_fft', 'pytorch', 'pytorch_fft']
# hard-coded expected energy levels for the different reconstruction modes
test_params_sparsity = [
    # increasing sparsity, no inhibition
    {'fit': dict(sparsity_H=0.0), 'nmf': dict(), 'res': dict(energy=186.666013, norm_H_1=7704.38977, norm_H_0=176346)},
    {'fit': dict(sparsity_H=0.1), 'nmf': dict(), 'res': dict(energy=225.494731, norm_H_1=6563.91176, norm_H_0=174037)},
    {'fit': dict(sparsity_H=0.5), 'nmf': dict(), 'res': dict(energy=858.621063, norm_H_1=4258.82541, norm_H_0=155247)},
    {'fit': dict(sparsity_H=1.0), 'nmf': dict(), 'res': dict(energy=2429.69334, norm_H_1=2114.50047, norm_H_0=136396)},
    {'fit': dict(sparsity_H=5.0), 'nmf': dict(), 'res': dict(energy=5351.91865, norm_H_1=3.0800e-06, norm_H_0=65338)},
    {'fit': dict(sparsity_H=10.), 'nmf': dict(), 'res': dict(energy=5351.91866, norm_H_1=2.5103e-13, norm_H_0=62486)},
]
test_params_inhibition = [
    # no sparsity, increasing inhibition
    {'fit': dict(inhibition_strength=0.1), 'nmf': dict(), 'res': dict(energy=435.993252, norm_H_1=5574.6769, norm_H_0=175483)},
    {'fit': dict(inhibition_strength=0.5), 'nmf': dict(), 'res': dict(energy=1831.92669, norm_H_1=3031.4130, norm_H_0=168931)},
    {'fit': dict(inhibition_strength=1.0), 'nmf': dict(), 'res': dict(energy=2665.70594, norm_H_1=2252.1331, norm_H_0=160644)},
    {'fit': dict(inhibition_strength=5.0), 'nmf': dict(), 'res': dict(energy=3618.33671, norm_H_1=1947.1969, norm_H_0=129929)},
    {'fit': dict(inhibition_strength=10.), 'nmf': dict(), 'res': dict(energy=3779.64954, norm_H_1=1926.9570, norm_H_0=118795)},
    # no sparsity, increasing inhibition, smaller inhibition range
    # pylint: disable=line-too-long
    {'fit': dict(inhibition_strength=0.1), 'nmf': dict(inhibition_range=((3, 3))), 'res': dict(energy=234.838968, norm_H_1=6730.89543, norm_H_0=176347)},  # noqa: E501
    {'fit': dict(inhibition_strength=0.5), 'nmf': dict(inhibition_range=((3, 3))), 'res': dict(energy=680.585424, norm_H_1=5177.87844, norm_H_0=174277)},  # noqa: E501
    {'fit': dict(inhibition_strength=1.0), 'nmf': dict(inhibition_range=((3, 3))), 'res': dict(energy=1119.00855, norm_H_1=4657.19574, norm_H_0=168777)},  # noqa: E501
    {'fit': dict(inhibition_strength=5.0), 'nmf': dict(inhibition_range=((3, 3))), 'res': dict(energy=518.936361, norm_H_1=6872.57858, norm_H_0=100488)},  # noqa: E501
    {'fit': dict(inhibition_strength=10.), 'nmf': dict(inhibition_range=((3, 3))), 'res': dict(energy=489.935256, norm_H_1=7224.76002, norm_H_0=62017)},  # noqa: E501
]

# create the input by concatenating the test image twice
img = racoon_image(gray=False, scale=0.1)
V = np.repeat(img.transpose((2, 0, 1))[np.newaxis, ...], 2, axis=0)


def _do_test(backend, params):
    # use the same random seed for all runs
    np.random.seed(seed=42)

    # create and fit the NMF model
    nmf = TransformInvariantNMF(
        n_atoms=10,
        atom_shape=(7, 7),
        n_iterations=25,
        backend=backend,
        verbose=3,
        **params['nmf']
    )
    nmf.fit(V, **params['fit'])

    H = nmf.H

    energy = nmf._energy_function(V)  # pylint: disable=protected-access
    norm_H_1 = np.sum(np.abs(H))
    norm_H_0 = np.sum(H/H.max() > 1e-7)

    nmf._logger.debug(f'energy={energy}, norm_H_1={norm_H_1}, norm_H_0={norm_H_0}')  # pylint: disable=protected-access

    expectation = params['res']
    assert np.isclose(energy, expectation['energy'])
    assert np.isclose(norm_H_1, expectation['norm_H_1'])
    assert np.isclose(norm_H_0, expectation['norm_H_0'])


@pytest.mark.parametrize('params', test_params_sparsity)
@pytest.mark.parametrize('backend', backends)
def test_sparsity(backend: str, params: Dict):
    _do_test(backend, params)


@pytest.mark.parametrize('params', test_params_inhibition)
@pytest.mark.parametrize('backend', backends)
def test_inhibition(backend: str, params: Dict):
    _do_test(backend, params)
