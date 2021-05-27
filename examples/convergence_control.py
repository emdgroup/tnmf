"""
===============================================================
Convergence control and iteration abort methods
===============================================================
This examples demonstrates how to log and plot the reconstruction
energy and use it's value to abort iteration if a convergence
criterion has been reched.
"""

import logging

import numpy as np
import matplotlib.pyplot as plt

from tnmf.TransformInvariantNMF import TransformInvariantNMF
from tnmf.utils.data_loading import racoon_image

print(__doc__)

# activate logging output to ensure, fitting progress is printed to the command line
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

# load an example image
img = racoon_image(scale=0.1)

# Construct a TransformInvariantNMF instance with selected parameters for the model.
n_atoms = 16
nmf = TransformInvariantNMF(
    n_atoms=n_atoms,
    atom_shape=(6, 10),
    n_iterations=500,
    reconstruction_mode='circular',  # valid, full, circular, reflect
    backend='pytorch',
    logger=None,
    verbose=3,
)

# Samples provided to nmf.fit() have to be indexed as V[sample, channel, sample_dim_1 .. sample_dim_n].
# This example uses one single-channel image.
img = img[np.newaxis, np.newaxis, ...]

# Define a progress_callback to take note of the reconstruction energy in every iteration.
reconstruction_energy = list()


def progress_callback(nmf_instance: TransformInvariantNMF, iteration: int) -> bool:
    energy = nmf_instance._energy_function(img)  # TODO: having this function protected is impractical
    reconstruction_energy.append([iteration, energy])

    # Continue iteration as long as energy is above a certain threshhold.
    return energy > 20.


# Run the fitting, i. e. compute dictionary W and activations H so that img = W*H
# Note that setting a progress_callback suppresses regular convergence output.
nmf.fit(img, inhibition_strength=0.0, progress_callback=progress_callback)

# Collect results from the TransformInvariantNMF instance.
img_r = nmf.R

# Create a plot of the original image and the reconstruction.
fig, axes = plt.subplots(nrows=1, ncols=2, squeeze=False, figsize=(8, 4))

for (data, title, kwargs), ax in zip([
        (img[0, 0], 'Original', dict(cmap='Greys')),
        (img_r[0, 0], 'Reconstruction', dict(cmap='Greys')),
        ], axes.flatten()):
    ax.set_title(title)
    ax.imshow(data, **kwargs, vmin=0., vmax=1.)

plt.tight_layout()

# Create a plot of the reconstruction energy over iterations.
reconstruction_energy = np.array(reconstruction_energy).T
plt.figure(figsize=(6, 4))
plt.plot(reconstruction_energy[0], reconstruction_energy[1])
plt.xlabel('Iteration')
plt.ylabel('Reconstruction Energy')

plt.show()
