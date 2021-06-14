"""
=================================================================
Decomposition of a gray scale image into characteristic patches
=================================================================
This examples demonstrates how to decompose a greyscale image
into a number of characteristic rectangular patches.

Lateral inhibition is used to promote localized activations
while circular reconstructions ensures to avoid boundary effects
that would be visible in the atoms and the reconstruction.
"""
import logging

import numpy as np
import matplotlib.pyplot as plt

from tnmf.TransformInvariantNMF import TransformInvariantNMF
from tnmf.utils.data_loading import racoon_image

print(__doc__)

# Activate logging output to ensure fitting progress is printed to the command line.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

# Load an example image.
img = racoon_image(scale=0.1)

# Construct a TransformInvariantNMF instance with selected parameters for the model.
n_atoms = 16
nmf = TransformInvariantNMF(
    n_atoms=n_atoms,
    atom_shape=(6, 10),
    n_iterations=500,
    reconstruction_mode='circular',
    backend='pytorch',
    verbose=3,
)

# Samples provided to nmf.fit() have to be indexed as V[sample, channel, sample_dim_1 .. sample_dim_n].
# This example uses one single-channel image.
img = img[np.newaxis, np.newaxis, ...]

# Run the fitting, i.e. compute dictionary W and activations H so that img = H*W.
nmf.fit(img, inhibition_strength=0.05)

# Collect results from the TransformInvariantNMF instance.
img_r = nmf.R
img_r_partial = np.array([nmf.R_partial(i_atom) for i_atom in range(n_atoms)])
dictionary = nmf.W
activations = nmf.H

# Create a plot of the original image and the reconstruction.
fig, axes = plt.subplots(ncols=2, squeeze=False, figsize=(8, 4))
for (data, title, kwargs), ax in zip([
        (img[0, 0], 'Original', dict(cmap='Greys')),
        (img_r[0, 0], 'Reconstruction', dict(cmap='Greys')),
        ], axes.flatten()):
    ax.set_title(title)
    ax.imshow(data, **kwargs, vmin=0., vmax=1.)
plt.tight_layout()
plt.show()

# Create a plot of the dictionary atoms W, the respective activations H and partial reconstructions.
for data, title, kwargs in [
        (dictionary[:, 0], 'Dictionary Atoms W', dict(cmap='Greys')),
        (activations[0, :], 'Atom Activations H', dict(cmap='gist_ncar')),
        (img_r_partial[:, 0, 0], 'Partial Reconstructions', dict(cmap='Greys')),
]:
    plot_rows = 4
    fig, axes = plt.subplots(nrows=plot_rows, ncols=n_atoms // plot_rows, squeeze=False, figsize=(8, 8),
                             subplot_kw=dict(xticks=[], yticks=[]))
    fig.suptitle(title)
    for i_atom, ax in enumerate(axes.flatten()):
        ax.set_title(i_atom)
        ax.imshow(data[i_atom], **kwargs)
    plt.tight_layout()
    plt.show()
