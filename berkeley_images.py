"""
Author: Mathias Winkel
"""

import os
import glob
import imageio
import logging

import numpy as np
import matplotlib.pyplot as plt
from TransformInvariantNMF import SparseNMF, ShiftInvariantNMF

COLOR_SELECTIONS = {
    'grey': lambda img: [np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])[:, :, np.newaxis]],
    'red': lambda img: [img[..., 0:1]],
    'green': lambda img: [img[..., 1:2]],
    'blue': lambda img: [img[..., 2:3]],
    'color basis': lambda img: [img],
    'colors (identical basis)': lambda img: [img[..., 0:1], img[..., 1:2], img[..., 2:3]],
}


def load_images(path: str, pattern: str, max_images: int = 0,
                color_mode: str = 'grey', dtype=np.float32) -> (np.ndarray, tuple):
    images = []
    rows, cols = 10000, 10000

    logging.info(f'Loading files from "{path}" with pattern "{pattern}" in color mode "{color_mode}"')

    count = 0
    for count, filename in enumerate(glob.glob(os.path.join(os.getcwd(), path, pattern))):
        # limit number of images
        if 0 < max_images <= count:
            break

        logging.info(filename)

        img = (imageio.imread(filename)[:, :, :3] / 255.).astype(dtype)
        channels = COLOR_SELECTIONS[color_mode](img)

        # flip all images to landscape orientation
        channels = [np.swapaxes(ch, 0, 1) if ch.shape[1] < ch.shape[0] else ch for ch in channels]

        for channel in channels:
            rows, cols = min(rows, channel.shape[0]), min(cols, channel.shape[1])
            images.append(channel)

    # cut all images to fit the smallest image
    images = [image[:rows, :cols, :] for image in images]

    images = np.asarray(images, dtype=dtype)

    # roll image index to the last dimension because this is how the NMF needs its data
    images = np.moveaxis(images, 0, -1)

    # just ensure nothing was garbled in the code above
    assert (images.ndim == 4)
    assert (images.shape[0] == rows and images.shape[1] == cols)
    assert (images.shape[2] in (1, 3))
    assert (images.shape[3] in (count, 3 * count))
    assert (images.dtype == dtype)

    return images, (rows, cols)


def compute_nmf(V, nmf_params):
    """Streamlit caching of NMF fitting."""
    nmf_params = nmf_params.copy()
    if nmf_params.pop('shift_invariant'):
        nmf = ShiftInvariantNMF(**nmf_params)
    else:
        nmf = SparseNMF(**nmf_params)
    nmf.fit(V)
    return nmf


def plot_signal_reconstruction(nmf, signal_number, samples_per_image):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8), subplot_kw=dict(xticks=[], yticks=[], frame_on=False))
    for iplot, ax in enumerate(axes):
        if iplot == 0:
            data = np.squeeze(nmf.V[..., signal_number * samples_per_image:(signal_number + 1) * samples_per_image])
        else:
            data = np.clip(np.squeeze(nmf.R[..., signal_number * samples_per_image:(signal_number + 1) * samples_per_image]), 0., 1.)

        ax.imshow(data)
        ax.grid(False)

    return [fig]


def plot_dictionary(W, num_columns=5):
    nrows = (W.shape[-1] + num_columns - 1) // num_columns
    fig = plt.figure(figsize=(2*num_columns, 2*nrows))

    for m in range(W.shape[-1]):
        ax = fig.add_subplot(nrows, num_columns, m+1, xticks=[], yticks=[])
        im = ax.imshow(W[..., m])
        ax.set_title(f'{m}')
        ax.grid(False)
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    return [fig]


def plot_activations(H, nmf_params, signal_number, samples_per_image):

    cm = ['Greys'] if samples_per_image == 1 else ['Reds', 'Greens', 'Blues']

    figs = []
    for atom in range(H.shape[-2]):
        fig, axes = plt.subplots(nrows=1, ncols=samples_per_image, squeeze=False, figsize=(samples_per_image*10, 5),
                                 subplot_kw=dict(xticks=[], yticks=[], frame_on=False))

        for channel, ax in zip(range(samples_per_image), axes.flatten()):
            if nmf_params['shift_invariant']:
                im = ax.imshow(np.squeeze(H[..., atom, signal_number * samples_per_image + channel]), aspect='equal',
                               cmap=cm[channel])
            else:
                im = ax.imshow(H[0], aspect='equal')
            ax.set_title(f'atom {atom}, channel {channel}')
            plt.colorbar(im, ax=ax)

        plt.tight_layout()
        figs.append(fig)

    return figs


def plot_partial_reconstruction(nmf, nmf_params, signal_number, samples_per_image):
    # TODO: should be possible to merge this with plot_activations()
    cm = ['Greys'] if samples_per_image == 1 else ['Reds', 'Greens', 'Blues']

    figs = []
    for atom in range(nmf.n_components):
        fig, axes = plt.subplots(nrows=1, ncols=samples_per_image, squeeze=False, figsize=(samples_per_image*10, 5))

        for channel, ax in zip(range(samples_per_image), axes.flatten()):
            if nmf_params['shift_invariant']:
                H_partial = nmf.partial_reconstruct(signal_number * samples_per_image + channel, 0, atom)
                im = ax.imshow(np.squeeze(H_partial), aspect='equal', cmap=cm[channel], vmin=0., vmax=1.)
            else:
                raise NotImplementedError
            ax.set_title(f'atom {atom}, channel {channel}')
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax)

        figs.append(fig)

    return figs


def close_figs(figs):
    if isinstance(figs, list):
        for fig in figs:
            close_figs(fig)
    else:
        plt.close(figs)


def st_plot(title, figs):
    for ifig, fig in enumerate(figs):
        filename = f'{title}_{ifig}.png'
        fig.savefig(filename)
    close_figs(figs)


if __name__ == '__main__':
    # -------------------- settings -------------------- #

    np.random.seed(42)

    d = r'BSR_bsds500/BSR/BSDS500/data/images/train'
    f = '*.jpg'
    max_images = 5
    color_mode = 'colors (identical basis)'
    dtype = np.float64

    images, image_shape = load_images(d, f, max_images, color_mode, dtype=dtype)

    nmf_params = {
        'verbose': 2,
        'use_fft': True,
        'shift_invariant': True,
        'sparsity_H': 0.5,
        'n_iterations': 1000,
        'refit_H': True,
        'n_components': 16,
        'atom_size': 8,
        'inhibition_range': None,
        'inhibition_strength': 0.25,
    }

    logging.info(f'NMF params: {nmf_params}')

    # -------------------- model fitting -------------------- #

    # fit the NMF model
    nmf = compute_nmf(images, nmf_params)

    # -------------------- visualization -------------------- #

    st_plot(f'Learned dictionary - {color_mode}', plot_dictionary(nmf.W))

    # select signal to be visualized
    samples_per_image = 3 if color_mode == 'colors (identical basis)' else 1

    signal_number = 0 # min_value=0, max_value=nmf.n_signals//samples_per_image - 1

    st_plot('Comparison to original signal',
            plot_signal_reconstruction(nmf, signal_number, samples_per_image))

    st_plot('Activations',
            plot_activations(nmf.H, nmf_params, signal_number, samples_per_image))

    st_plot('Partial Reconstructions',
            plot_partial_reconstruction(nmf, nmf_params, signal_number, samples_per_image))
