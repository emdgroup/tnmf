"""
=================================================================
Convergence of the MiniBatch Algorithms
=================================================================
This examples compares, for a number of one-dimensional signals
from an ECG data, the convergence speed for the default multiplica-
tive update and different mini batch algorithms versus used
iterations/epochs and elapsed wall clock time.
To this end, the fit procedure is run until a pre-defined
level of convergence is achieved.

Attention: This example has a rather long runtime.
"""
from itertools import cycle, product
from time import process_time

import numpy as np
from scipy import signal
from scipy.misc import electrocardiogram as ecg
import matplotlib.pyplot as plt

from tnmf.TransformInvariantNMF import TransformInvariantNMF, MiniBatchAlgorithm

print(__doc__)

# Load some example data
V = ecg()
# Low-pass filter the ECG data to remove slowly moving offset
V = signal.filtfilt(*signal.butter(3, 0.05, 'highpass'), V)
# need non-negative data
V -= V.min()
# split the 1D curve into 100 individual samples
V = V.reshape((-1, 360*3))

# Samples provided to nmf.fit() have to be indexed as V[sample, channel, sample_dim_1 .. sample_dim_n].
# This example uses multiple one-dimensional single channel data series.
V = V[:, np.newaxis, ...]


def do_fit(
    v,
    inhibition_strength=0.01,
    sparsity_H=0.01,
    **kwargs,
):
    # use the same random seed for all runs
    np.random.seed(seed=42)

    # Define a progress callback to keep track of the reconstruction energy in every iteration.
    reconstruction_energy = list()

    def progress_callback(nmf_instance: TransformInvariantNMF, iteration: int) -> bool:
        e = nmf_instance._energy_function()
        print(f'Iteration: {iteration}, Reconstruction Energy: {e:.2f}', end='\r')
        reconstruction_energy.append([iteration, e])
        # Continue iteration as long as energy is above a certain threshold.
        return e > 300

    # Construct a TransformInvariantNMF instance with selected parameters for the model.
    nmf = TransformInvariantNMF(
        n_atoms=9,
        atom_shape=(100, ),
        reconstruction_mode='valid',
        backend='numpy_caching_fft',
        verbose=3,
    )

    t = -process_time()
    # Run the fitting, i.e. compute dictionary W and activations H so that V = H*W.
    # Note that setting a progress callback suppresses regular convergence output.
    nmf.fit(
        v,
        inhibition_strength=inhibition_strength,
        sparsity_H=sparsity_H,
        progress_callback=progress_callback,
        **kwargs)
    t += process_time()

    print(f'\nFinished after {t:.2f} seconds.')

    # Collect results from the TransformInvariantNMF instance.
    return np.asarray(reconstruction_energy).T, nmf.R, nmf.W, t


results = {}

max_iter = 100

linestyles = (['-', '--', '-.', ':'], ['b', 'g', 'r', 'c', 'm', 'k'])
linestyles1 = cycle(product(*linestyles))
linestyles2 = cycle(product(*linestyles))

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16, 8))
fig.suptitle('Reconstruction Energy')
axes[0].set_xlabel('Iteration / Epoch')
axes[1].set_xlabel('Time / s')

for params in (
    dict(),
    #
    dict(algorithm=MiniBatchAlgorithm.Cyclic_MU, batch_size=10),
    dict(algorithm=MiniBatchAlgorithm.ASG_MU, batch_size=10),
    dict(algorithm=MiniBatchAlgorithm.GSG_MU, batch_size=10),
    dict(algorithm=MiniBatchAlgorithm.ASAG_MU, batch_size=10),
    dict(algorithm=MiniBatchAlgorithm.GSAG_MU, batch_size=10),
    #
    dict(algorithm=MiniBatchAlgorithm.ASG_MU, batch_size=1),
    dict(algorithm=MiniBatchAlgorithm.ASG_MU, batch_size=5),
    dict(algorithm=MiniBatchAlgorithm.ASG_MU, batch_size=20),
    dict(algorithm=MiniBatchAlgorithm.ASG_MU, batch_size=40),
    dict(algorithm=MiniBatchAlgorithm.ASG_MU, batch_size=100),
):
    sp = str(params)
    print(f'Working on {sp}')
    energy, R, W, dt = do_fit(V, **params)

    # plot reconstruction energy vs total wall time
    axes[1].plot(np.linspace(0, dt, len(energy[1])), energy[1], ''.join(next(linestyles2)), label=sp, linewidth=1.)
    # plot reconstruction energy vs iteration/epoch
    axes[0].plot(energy[0], energy[1], ''.join(next(linestyles1)), label=sp, linewidth=1.)

plt.legend()
plt.show()
